import collections
import copy
import datetime
import time
from time import sleep
from tracemalloc import start

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_cutting import (cut_wires, expand_observables,
                                  generate_cutting_experiments,
                                  partition_problem,
                                  reconstruct_expectation_values)
from qiskit_addon_cutting.automated_cut_finding import (DeviceConstraints,
                                                        OptimizationParameters,
                                                        find_cuts)
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Batch, SamplerV2

from engine.metrics.csv_writer import MetricsFileWriter
from mini_apps.quantum_simulation.motifs.base_motif import Motif
from mini_apps.quantum_simulation.motifs.qiskit_benchmark import generate_data
from qiskit.primitives import (
    SamplerResult,  # for SamplerV1
    PrimitiveResult,  # for SamplerV2
)

from qiskit_aer.primitives import EstimatorV2
import logging


DEFAULT_SIMULATOR_BACKEND_OPTIONS = {"backend_options": {"device":"CPU", "method": "statevector"}}


def execute_sampler(sampler, label, subsystem_subexpts, shots):
    submit_start = time.time()
    job = sampler.run(subsystem_subexpts, shots=shots)
    submit_end = time.time()
    result_start = time.time()
    result = job.result()    
    result_end = time.time()
    print(f"Job {label} completed with job id {job.job_id()}, submit_time: {submit_end-submit_start} and execution_time: {result_end - result_start}, type: {type(result)}")
    return (label, result)

def run_full_circuit(observable, backend_options, full_circuit_transpilation):
    estimator = EstimatorV2(options=backend_options)
    exact_expval = estimator.run([(full_circuit_transpilation, observable)]).result()[0].data.evs
    return exact_expval


class CircuitCuttingBuilder:
    def __init__(self):
        self.subcircuit_size = None
        self.base_qubits = None
        self.observables = None
        self.scale_factor = None
        self.qiskit_backend_options = None
        self.sub_circuit_task_resources = {'num_cpus': 1, 'num_gpus': 0, 'memory': None}

    def set_subcircuit_size(self, subcircuit_size):
        self.subcircuit_size = subcircuit_size
        return self

    def set_base_qubits(self, base_qubits):
        self.base_qubits = base_qubits
        return self

    def set_observables(self, observables):
        self.observables = observables
        return self

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor
        return self

    def set_result_file(self, result_file):
        self.result_file = result_file
        return self
    
    def set_qiskit_backend_options(self, qiskit_backend_options):
        self.qiskit_backend_options = qiskit_backend_options
        return self
    
    def set_sub_circuit_task_resources(self, sub_circuit_task_resources):
        self.sub_circuit_task_resources = sub_circuit_task_resources
        return self

    def set_full_circuit_task_resources(self, full_circuit_task_resources):
        self.full_circuit_task_resources = full_circuit_task_resources
        return self    
    
    def build(self, executor):
        return CircuitCutting(executor, self.subcircuit_size, self.base_qubits, self.observables, self.scale_factor, self.qiskit_backend_options, self.sub_circuit_task_resources, self.full_circuit_task_resources, self.result_file)




class CircuitCutting(Motif):
    def __init__(self, executor, subcircuit_size, base_qubits, observables, scale_factor, qiskit_backend_options, sub_circuit_task_resources ,full_circuit_task_resources, result_file):
        super().__init__(executor, base_qubits)
        self.subcircuit_size = subcircuit_size
        self.observables = observables
        self.scale_factor = scale_factor
        self.result_file = result_file
        self.qiskit_backend_options = qiskit_backend_options
        self.base_qubits = base_qubits
        self.experiment_start_time = datetime.datetime.now()
        self.sub_circuit_task_resources = sub_circuit_task_resources
        self.full_circuit_task_resources = full_circuit_task_resources
        header = ["experiment_start_time", "subcircuit_size", "base_qubits", "observables", "scale_factor", 
                  "transpile_time_secs", "subcircuit_exec_time_secs", "reconstruct_subcircuit_expectations_time_secs", "full_circuit_estimator_runtime", "error_in_estimation"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)
        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create a console handler and set the log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the console handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)
        
        self.logger = logger

    def pre_processing(self):    
        circuit = EfficientSU2(self.base_qubits * self.scale_factor, entanglement="linear", reps=2).decompose()
        circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
        
        observable = SparsePauliOp([o * self.scale_factor for o in self.observables])


        # Specify settings for the cut-finding optimizer
        optimization_settings = OptimizationParameters(seed=111)

        # Specify the size of the QPUs available
        device_constraints = DeviceConstraints(qubits_per_subcircuit=self.subcircuit_size)

        cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
        self.logger.info(
            f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
            f'overhead of {metadata["sampling_overhead"]}.\n'
            f'Lowest cost solution found: {metadata["minimum_reached"]}.'
        )
        for cut in metadata["cuts"]:
            self.logger.info(f"{cut[0]} at circuit instruction index {cut[1]}")


        qc_w_ancilla = cut_wires(cut_circuit)
        observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)

        partitioned_problem = partition_problem(
            circuit=qc_w_ancilla, observables=observables_expanded
        )
        subcircuits = partitioned_problem.subcircuits
        subobservables = partitioned_problem.subobservables
        self.logger.info(
            f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}"
        )

        subexperiments, coefficients = generate_cutting_experiments(
            circuits=subcircuits, observables=subobservables, num_samples=1_000
        )
        self.logger.info(
            f"{len(subexperiments[0]) + len(subexperiments[1])} total subexperiments to run on backend."
        )
        return subexperiments, coefficients, subobservables, observable, circuit

            
    def run(self):
        subexperiments, coefficients, subobservables, observable, circuit = self.pre_processing()
        
        backend_options = DEFAULT_SIMULATOR_BACKEND_OPTIONS
        if self.qiskit_backend_options:
            backend_options = self.qiskit_backend_options
        self.logger.info(f"Backend options: {backend_options}")
        backend = AerSimulator(**backend_options["backend_options"])
        
        traspile_start_time = time.time()
        self.logger.info("*********************************** transpiling circuits ***********************************")
        # Transpile the subexperiments to ISA circuits
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_subexperiments = {}
        for label, partition_subexpts in subexperiments.items():
            isa_subexperiments[label] = pass_manager.run(partition_subexpts)
        self.logger.info("*********************************** transpiling done ***********************************")
        traspile_end_time = time.time()
        transpile_time_secs = traspile_end_time - traspile_start_time
        self.logger.info(f"Transpile time: {transpile_time_secs}")
        
    
        tasks=[]
        i=0
        sub_circuit_execution_time = time.time()
        resources = copy.copy(self.sub_circuit_task_resources)
        with Batch(backend=backend) as batch:
            sampler = SamplerV2(mode=batch)
            self.logger.info(f"*********************************** len of subexperiments {len(isa_subexperiments)}*************************")
            for label, subsystem_subexpts in isa_subexperiments.items():
                self.logger.info(len(subsystem_subexpts))
                task_future = self.executor.submit_task(execute_sampler, sampler, label, subsystem_subexpts, resources=resources, shots=2**12)
                tasks.append(task_future)
                i=i+1

        results_tuple=self.executor.get_results(tasks)
        sub_circuit_execution_end_time = time.time()
        subcircuit_exec_time_secs = sub_circuit_execution_end_time - sub_circuit_execution_time
        self.logger.info(f"Execution time for subcircuits: {subcircuit_exec_time_secs}")
        
        
        # Get all samplePubResults            
        samplePubResults = collections.defaultdict(list)
        for result in results_tuple:
            print(result)
            self.logger.info(f"Result: {result[0], result[1]}")
            samplePubResults[result[0]].extend(result[1]._pub_results)        
        
        results = {}
        for label, samples in samplePubResults.items():
            results[label] = PrimitiveResult(samples)

        
        reconstruction_start_time = time.time()
        # Get expectation values for each observable term
        reconstructed_expvals = reconstruct_expectation_values(
            results,
            coefficients,
            subobservables,
        )
        reconstruction_end_time = time.time()
        reconstruct_subcircuit_expectations_time_secs = reconstruction_end_time-reconstruction_start_time
        self.logger.info(f"Execution time for reconstruction: {reconstruct_subcircuit_expectations_time_secs}")
        
        final_expval = np.dot(reconstructed_expvals, observable.coeffs)   
        
        
        exact_expval = 0
        transpile_full_circuit_time = time.time()
        full_circuit_transpilation = pass_manager.run(circuit)
        transpile_full_circuit_end_time = time.time()
        self.logger.info(f"Execution time for full Circuit transpilation: {transpile_full_circuit_end_time-transpile_full_circuit_time}")
                
        
        full_circuit_estimator_time = time.time()                           
        full_circuit_task = self.executor.submit_task(run_full_circuit, observable, backend_options, full_circuit_transpilation, resources=self.full_circuit_task_resources)
        exact_expval = self.executor.get_results([full_circuit_task])
        full_circuit_estimator_runtime = time.time()-full_circuit_estimator_time
        
        self.logger.info(f"Execution time for full Circuit: {full_circuit_estimator_runtime}")         
        self.logger.info(f"Reconstructed expectation value: {np.real(np.round(final_expval, 8))}")
        self.logger.info(f"Exact expectation value: {np.round(exact_expval, 8)}")
        error_in_estimation=np.real(np.round(final_expval-exact_expval, 8))
        self.logger.info(f"Error in estimation: {error_in_estimation}")
        self.logger.info(
            f"Relative error in estimation: {np.real(np.round((final_expval-exact_expval) / exact_expval, 8))}"
        )                    

        self.metrics_file_writer.write([self.experiment_start_time, self.subcircuit_size, self.base_qubits, 
                                        self.observables, self.scale_factor, transpile_time_secs, subcircuit_exec_time_secs, reconstruct_subcircuit_expectations_time_secs, full_circuit_estimator_runtime,error_in_estimation])

        self.metrics_file_writer.close()


SUBCIRCUIT_SIZE = "subcircuit_size"
BASE_QUBITS = "base_qubits"
OBSERVABLES= "observables"
SCALE_FACTOR= "scale_factor"
SUB_CIRCUIT_TASK_RESOURCES = "sub_circuit_task_resources"
FULL_CIRCUIT_TASK_RESOURCES = "full_circuit_task_resources"
SIMULATOR_BACKEND_OPTIONS = "simulator_backend_options"
