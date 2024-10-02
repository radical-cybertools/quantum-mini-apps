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


def execute_sampler(sampler, label, subsystem_subexpts, shots):
    submit_start = time.time()
    job = sampler.run(subsystem_subexpts, shots=shots)
    submit_end = time.time()
    result_start = time.time()
    result = job.result()    
    result_end = time.time()
    print(f"Job {label} completed with job id {job.job_id()}, submit_time: {submit_end-submit_start} and execution_time: {result_end - result_start}, type: {type(result)}")
    return (label, result)


class CircuitCuttingBuilder:
    def __init__(self):
        self.subcircuit_size = None
        self.base_qubits = None
        self.observables = None
        self.scale_factor = None
        self.qiskit_backend_options = {"method": "statevector"}
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
    
    def build(self, executor):
        return CircuitCutting(executor, self.subcircuit_size, self.base_qubits, self.observables, self.scale_factor, self.qiskit_backend_options, self.sub_circuit_task_resources, self.result_file)




class CircuitCutting(Motif):
    def __init__(self, executor, subcircuit_size, base_qubits, observables, scale_factor, qiskit_backend_options, sub_circuit_task_resources , result_file):
        super().__init__(executor, base_qubits)
        self.subcircuit_size = subcircuit_size
        self.observables = observables
        self.scale_factor = scale_factor
        self.result_file = result_file
        self.qiskit_backend_options = qiskit_backend_options
        self.base_qubits = base_qubits
        self.experiment_start_time = datetime.datetime.now()
        self.sub_circuit_task_resources = sub_circuit_task_resources
        header = ["experiment_start_time", "subcircuit_size", "base_qubits", "observables", "scale_factor", 
                  "transpile_time_secs", "subcircuit_exec_time_secs", "reconstruct_subcircuit_expectations_time_secs", "total_runtime_secs"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)

    def pre_processing(self):    
        circuit = EfficientSU2(self.base_qubits * self.scale_factor, entanglement="linear", reps=2).decompose()
        circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
        
        observable = SparsePauliOp([o * self.scale_factor for o in self.observables])


        # Specify settings for the cut-finding optimizer
        optimization_settings = OptimizationParameters(seed=111)

        # Specify the size of the QPUs available
        device_constraints = DeviceConstraints(qubits_per_subcircuit=self.subcircuit_size)

        cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
        print(
            f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
            f'overhead of {metadata["sampling_overhead"]}.\n'
            f'Lowest cost solution found: {metadata["minimum_reached"]}.'
        )
        for cut in metadata["cuts"]:
            print(f"{cut[0]} at circuit instruction index {cut[1]}")


        qc_w_ancilla = cut_wires(cut_circuit)
        observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)

        partitioned_problem = partition_problem(
            circuit=qc_w_ancilla, observables=observables_expanded
        )
        subcircuits = partitioned_problem.subcircuits
        subobservables = partitioned_problem.subobservables
        print(
            f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}"
        )

        subexperiments, coefficients = generate_cutting_experiments(
            circuits=subcircuits, observables=subobservables, num_samples=1_000
        )
        print(
            f"{len(subexperiments[0]) + len(subexperiments[1])} total subexperiments to run on backend."
        )
        return subexperiments, coefficients, subobservables, observable, circuit

            
    def run(self):
        total_start_time = time.time()
        subexperiments, coefficients, subobservables, observable, circuit = self.pre_processing()
        
        backend = AerSimulator(device=self.qiskit_backend_options.get("device", "CPU"))
        
        traspile_start_time = time.time()
        print("*********************************** transpiling circuits ***********************************")
        # Transpile the subexperiments to ISA circuits
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_subexperiments = {}
        for label, partition_subexpts in subexperiments.items():
            isa_subexperiments[label] = pass_manager.run(partition_subexpts)
        print("*********************************** transpiling done ***********************************")
        traspile_end_time = time.time()
        transpile_time_secs = traspile_end_time - traspile_start_time
        print("Transpile time: ", transpile_time_secs)
        
        

        tasks=[]
        i=0
        sub_circuit_execution_time = time.time()
        resources = copy.copy(self.sub_circuit_task_resources)
        with Batch(backend=backend) as batch:
            sampler = SamplerV2(mode=batch)
            print(f"*********************************** len of subexperiments {len(isa_subexperiments)}*************************")
            for label, subsystem_subexpts in isa_subexperiments.items():
                print(len(subsystem_subexpts))
                task_future = self.executor.submit_task(execute_sampler, sampler, label, subsystem_subexpts[0], shots=2**12)
                tasks.append(task_future)
                i=i+1

        results_tuple=self.executor.get_results(tasks)
        # print(results_tuple)
        sub_circuit_execution_end_time = time.time()
        subcircuit_exec_time_secs = sub_circuit_execution_end_time - sub_circuit_execution_time
        print("Execution time for subcircuits: ", subcircuit_exec_time_secs)
        
        results = {}
        
        for result in results_tuple:
            results[result[0]] = result[1]
        
        reconstruct_start_time = time.time()
        # Get expectation values for each observable term
        reconstructed_expvals = reconstruct_expectation_values(
            results,
            coefficients,
            subobservables,
        )
        
        final_expval = np.dot(reconstructed_expvals, observable.coeffs)
        reconstruct_end_time = time.time()
        reconstruct_subcircuit_expectations_time_secs = reconstruct_end_time - reconstruct_start_time
        print("Reconstruct time: ", reconstruct_subcircuit_expectations_time_secs)
        total_end_time = time.time()
        total_run_time_secs = total_end_time - total_start_time
        print("Total time: ", total_run_time_secs)

        self.metrics_file_writer.write([self.experiment_start_time, self.subcircuit_size, self.base_qubits, 
                                        self.observables, self.scale_factor, transpile_time_secs, subcircuit_exec_time_secs, reconstruct_subcircuit_expectations_time_secs, total_run_time_secs])

        self.metrics_file_writer.close()


SUBCIRCUIT_SIZE = "subcircuit_size"
BASE_QUBITS = "base_qubits"
OBSERVABLES= "observables"
SCALE_FACTOR= "scale_factor"
SUB_CIRCUIT_TASK_RESOURCES = "sub_circuit_task_resources"