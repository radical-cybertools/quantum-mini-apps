# Standard library imports
import collections
import copy
import datetime
import json
import logging
import time
from time import sleep
from tracemalloc import start
import subprocess
import os

# Third party imports
import fire
import numpy as np
import ray  # Add this import statement

# Qiskit imports
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import PrimitiveResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator
from qiskit_addon_cutting import (
    cut_wires,
    expand_observables,
    generate_cutting_experiments,
    partition_problem,
    reconstruct_expectation_values,
)
from qiskit_addon_cutting.automated_cut_finding import (
    DeviceConstraints,
    OptimizationParameters,
    find_cuts,
)
from qiskit_ibm_runtime import Batch, SamplerV2
from qiskit import qpy

# Local imports
from engine.metrics.csv_writer import MetricsFileWriter
from mini_apps.quantum_simulation.motifs.base_motif import Motif
from mini_apps.quantum_simulation.motifs.qiskit_benchmark import generate_data


# Configuration parameter keys
# Circuit parameters
SUBCIRCUIT_SIZE = "subcircuit_size"  # Size of subcircuits after cutting
BASE_QUBITS = "base_qubits"  # Number of base qubits in circuit
OBSERVABLES = "observables"  # Observable operators to measure
SCALE_FACTOR = "scale_factor"  # Scaling factor for circuit size
NUM_SAMPLES = "num_samples"  # Number of measurement samples

# Resource configuration
SUB_CIRCUIT_TASK_RESOURCES = "sub_circuit_task_resources"  # Resources for subcircuit tasks
FULL_CIRCUIT_TASK_RESOURCES = "full_circuit_task_resources"  # Resources for full circuit tasks

# Backend configuration
CIRCUIT_CUTTING_SIMULATOR_BACKEND_OPTIONS = "circuit_cutting_simulator_backend_options"  # Backend options for cut circuits
FULL_CIRCUIT_SIMULATOR_BACKEND_OPTIONS = "full_circuit_simulator_backend_options"  # Backend options for full circuit

# Execution mode flags
FULL_CIRCUIT_ONLY = "full_circuit_only"  # Run only full circuit simulation
CIRCUIT_CUTTING_ONLY = "circuit_cutting_only"  # Run only circuit cutting simulation
SCENARIO_LABEL = "scenario_label"  # Label for the simulation scenario

# Default backend configuration
DEFAULT_SIMULATOR_BACKEND_OPTIONS = {
    "backend_options": {
        "device": "CPU",  # Use CPU device
        "method": "statevector"  # Use statevector simulation method
    }
}

##################################################################################################
# Called from distributed executor, e.g., Ray or MPI

# Circuit Cutting Simulation
def execute_sampler(backend_options, label, subsystem_subexpts, shots):
    # Add error handling
    try:
        from qiskit_aer import AerSimulator
        submit_start = time.time()
        backend = AerSimulator(**backend_options["backend_options"])

        with Batch(backend=backend) as batch:
            sampler = SamplerV2(mode=batch)
            job = sampler.run(subsystem_subexpts, shots=shots)
            submit_end = time.time()
            result_start = time.time()
            result = job.result()
            result_end = time.time()

            # debug
            for pub_result in result:
                # Debugging statements to inspect pub_result
                print("Attributes of pub_result:", dir(pub_result))
                print("pub_result:", pub_result)
                # Break after first iteration for debugging
                break

            # Reconstruct the PrimitiveResult object to fix serialization issues with current Qiskit versions (at the time 1.3)
            # see https://github.com/Qiskit/qiskit/issues/12787
            from qiskit.primitives.containers import (
                PrimitiveResult,
                SamplerPubResult,
                DataBin,
                BitArray,
            )

            # Override DataBin class to fix serialization issues
            class CustomDataBin(DataBin):
                def __setattr__(self, name, value):
                    super().__init__()
                    self.__dict__[name] = value

            # Reconstruct the PrimitiveResult object to fix serialization issues
            new_results = []
            for pub_result in result:
                # Deep copy the metadata
                new_metadata = copy.deepcopy(pub_result.metadata)

                # Access the DataBin object
                data_bin = pub_result.data

                # Reconstruct DataBin
                new_data_bin_dict = {}

                # Explicitly copy 'observable_measurements'
                if hasattr(data_bin, "observable_measurements"):
                    observable_measurements = data_bin.observable_measurements
                    new_observable_array = np.copy(observable_measurements.array)
                    new_observable_bitarray = BitArray(
                        new_observable_array, observable_measurements.num_bits
                    )
                    new_data_bin_dict["observable_measurements"] = new_observable_bitarray

                # Explicitly copy 'qpd_measurements'
                if hasattr(data_bin, "qpd_measurements"):
                    qpd_measurements = data_bin.qpd_measurements
                    new_qpd_array = np.copy(qpd_measurements.array)
                    new_qpd_bitarray = BitArray(new_qpd_array, qpd_measurements.num_bits)
                    new_data_bin_dict["qpd_measurements"] = new_qpd_bitarray

                # Copy other attributes of DataBin (e.g., 'shape')
                if hasattr(data_bin, "shape"):
                    new_data_bin_dict["shape"] = copy.deepcopy(data_bin.shape)

                # Create a new DataBin instance
                new_data_bin = CustomDataBin(**new_data_bin_dict)
                # new_data_bin.__setattr__ = custom_setattr

                # Create a new SamplerPubResult
                new_pub_result = SamplerPubResult(data=new_data_bin, metadata=new_metadata)
                new_results.append(new_pub_result)

            # Create a new PrimitiveResult
            new_result = PrimitiveResult(
                new_results, metadata=copy.deepcopy(result.metadata)
            )

            print(
                f"Job {label} completed with job id {job.job_id()}, submit_time: {submit_end-submit_start} and execution_time: {result_end - result_start}, type: {type(new_result)}"
            )
            return (label, new_result)
    except Exception as e:
        logging.error(f"Error executing sampler: {str(e)}")
        raise


# Full Circuit Simulation
def run_full_circuit(observable, backend_options, full_circuit):
    try:
        from qiskit_aer.primitives import EstimatorV2
        from qiskit_aer import AerSimulator
        
        # Create simulator
        simulator = AerSimulator(**backend_options["backend_options"])
        
        # Create estimator using the simulator
        estimator = EstimatorV2.from_backend(simulator)
        
        # Run estimation
        result = estimator.run([(full_circuit, observable)])
        exact_expval = result.result()[0].data.evs
        
        return exact_expval
        
    except Exception as e:
        logging.error(f"Unexpected error in full circuit simulation: {str(e)}")
        return str(e)
        # raise

def load_circuit(circuit_file):
    """Load circuit from file (qpy)"""
    with open(circuit_file, 'rb') as fd:
        circuits = qpy.load(fd)
    return circuits[0] 

def load_observable(observable_file):
    """Load observable from npy file"""
    # Load NumPy file
    loaded_data = np.load(observable_file, allow_pickle=True)
    deserialized_observable = SparsePauliOp.from_list(loaded_data.tolist())
    return deserialized_observable

def cli_run_full_circuit(
    observable_file: str,
    backend_options_file: str,
    circuit_file: str
    ):
    """
    Run full circuit simulation from command line
    
    Args:
        observable_file: Path to npy file containing observable data
        backend_options_file: Path to JSON file containing backend options
        circuit_file: Path to qpy file containing quantum circuit
    
    Returns:
        float: Expectation value
    """
    # Load inputs from files
    observable = load_observable(observable_file)
    with open(backend_options_file, 'r') as f:
        backend_options = json.load(f)
    full_circuit = load_circuit(circuit_file)
    
    # Import and run the original function
    from mini_apps.quantum_simulation.motifs.circuit_cutting_motif import run_full_circuit
    result = run_full_circuit(observable, backend_options, full_circuit)
    
    # Convert numpy types to Python native types for JSON serialization
    if isinstance(result, np.ndarray):
        result = result.tolist()

    # print(f"{result}")    
    return result



#####################################################################################################

class CircuitCuttingBuilder:
    """
    Builder class for configuring and constructing `CircuitCutting` objects with customizable settings.
    
    Attributes:
        subcircuit_size (int): Defines the number of qubits in each subcircuit.
        base_qubits (list): Specifies the base qubits involved in the circuit.
        observables (list): Lists the observables to be measured during the simulation.
        scale_factor (float): Determines the scaling factor applied to the circuit parameters.
        full_circuit_qiskit_options (dict): Configuration options for the Qiskit backend.
        circuit_cutting_qiskit_options (dict): Configuration options for the Qiskit backend.
        full_circuit_only (bool): When set to True, executes only the full circuit without any cutting.
        circuit_cutting_only (bool): When set to True, enables only circuit cutting without full circuit execution.
        num_samples (int): Number of samples to be used in the simulation.
        sub_circuit_task_resources (dict): Specifies computational resources allocated for sub-circuit tasks, such as CPU, GPU, and memory.
        full_circuit_task_resources (dict): Specifies computational resources allocated for full-circuit tasks, including CPU, GPU, and memory.
        result_file (str): Path to the file where simulation results will be stored.
    
    Methods:
        set_subcircuit_size(subcircuit_size: int) -> CircuitCuttingBuilder:
            Sets the size of the subcircuits.
    
        set_base_qubits(base_qubits: list) -> CircuitCuttingBuilder:
            Defines the base qubits for the circuit.
    
        set_observables(observables: list) -> CircuitCuttingBuilder:
            Specifies the observables to measure during the simulation.
    
        set_scale_factor(scale_factor: float) -> CircuitCuttingBuilder:
            Sets the scaling factor for circuit parameters.
    
        set_result_file(result_file: str) -> CircuitCuttingBuilder:
            Defines the file path for storing simulation results.
    
        set_qiskit_backend_options(qiskit_backend_options: dict) -> CircuitCuttingBuilder:
            Configures options for the Qiskit backend.
    
        set_full_circuit_qiskit_options(full_circuit_qiskit_options: dict) -> CircuitCuttingBuilder:
            Configures Qiskit backend options specifically for the full circuit simulation.
    
        set_circuit_cutting_qiskit_options(circuit_cutting_qiskit_options: dict) -> CircuitCuttingBuilder:
            Configures Qiskit backend options specifically for circuit cutting simulations.
    
        set_num_samples(num_samples: int) -> CircuitCuttingBuilder:
            Sets the number of samples to be used in the simulation.
    
        set_sub_circuit_task_resources(sub_circuit_task_resources: dict) -> CircuitCuttingBuilder:
            Allocates computational resources for sub-circuit tasks.
    
        set_full_circuit_task_resources(full_circuit_task_resources: dict) -> CircuitCuttingBuilder:
            Allocates computational resources for full-circuit tasks.
    
        set_full_circuit_only(full_circuit_only: bool) -> CircuitCuttingBuilder:
            Enables or disables the execution of only the full circuit.
    
        set_circuit_cutting_only(circuit_cutting_only: bool) -> CircuitCuttingBuilder:
            Enables or disables the use of only circuit cutting.
    
        build(executor) -> CircuitCutting:
            Constructs and returns a `CircuitCutting` object based on the configured settings.
    """
    def __init__(self):
        self.subcircuit_size = None
        self.base_qubits = None
        self.observables = None
        self.scale_factor = None
        self.full_circuit_qiskit_options = None
        self.circuit_cutting_qiskit_options = None
        self.full_circuit_only = False
        self.circuit_cutting_only = False
        self.num_samples = 10
        self.sub_circuit_task_resources = {"num_cpus": 1, "num_gpus": 0, "memory": None}
        self.full_circuit_task_resources = {"num_cpus": 1, "num_gpus": 0, "memory": None}

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

    def set_full_circuit_qiskit_options(self, full_circuit_qiskit_options):
        self.full_circuit_qiskit_options = full_circuit_qiskit_options
        return self

    def set_circuit_cutting_qiskit_options(self, circuit_cutting_qiskit_options):
        self.circuit_cutting_qiskit_options = circuit_cutting_qiskit_options
        return self

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples
        return self

    def set_sub_circuit_task_resources(self, sub_circuit_task_resources):
        self.sub_circuit_task_resources = sub_circuit_task_resources
        return self

    def set_full_circuit_task_resources(self, full_circuit_task_resources):
        self.full_circuit_task_resources = full_circuit_task_resources
        return self

    def set_full_circuit_only(self, full_circuit_only):
        self.full_circuit_only = full_circuit_only
        return self

    def set_circuit_cutting_only(self, circuit_cutting_only):
        self.circuit_cutting_only = circuit_cutting_only
        return self

    def set_scenario_label(self, scenario_label):
        self.scenario_label = scenario_label
        return self

    def build(self, executor):
        return CircuitCutting(
            executor,
            self.subcircuit_size,
            self.base_qubits,
            self.observables,
            self.scale_factor,
            self.full_circuit_qiskit_options,
            self.circuit_cutting_qiskit_options,
            self.sub_circuit_task_resources,
            self.full_circuit_task_resources,
            self.full_circuit_only,
            self.circuit_cutting_only,
            self.result_file,
            self.num_samples,
            self.scenario_label
        )


class CircuitCutting(Motif):

    def __init__(
        self,
        executor,
        subcircuit_size,
        base_qubits,
        observables,
        scale_factor,
        full_circuit_qiskit_options,
        circuit_cutting_qiskit_options,
        sub_circuit_task_resources,
        full_circuit_task_resources,
        full_circuit_only,
        circuit_cutting_only,
        result_file,
        num_samples,
        scenario_label
    ):
        super().__init__(executor, base_qubits)
        self.subcircuit_size = subcircuit_size
        self.observables = observables
        self.scale_factor = scale_factor
        self.result_file = result_file
        self.full_circuit_qiskit_options = full_circuit_qiskit_options
        self.circuit_cutting_qiskit_options = circuit_cutting_qiskit_options
        self.base_qubits = base_qubits
        self.experiment_start_time = datetime.datetime.now()
        self.num_samples = num_samples
        self.sub_circuit_task_resources = sub_circuit_task_resources
        self.full_circuit_task_resources = full_circuit_task_resources
        self.full_circuit_only = full_circuit_only
        self.circuit_cutting_only = circuit_cutting_only
       
        self.scenario_label = scenario_label
        self.metadata = None
        header = [
            "experiment_start_time",
            "subcircuit_size",
            "base_qubits",
            "observables",
            "scale_factor",
            "num_samples",
            "number_of_tasks",
            "metadata",
            "cluster_config",
            "full_circuit_qiskit_options",
            "circuit_cutting_qiskit_options",
            "circuit_cutting_task_resources",
            "full_circuit_task_resources",
            "find_cuts_time",
            "circuit_cutting_transpile_time_secs",
            "circuit_cutting_exec_time_secs",
            "circuit_cutting_reconstruct_time_secs",
            "circuit_cutting_total_runtime_secs",
            "full_circuit_transpile_time_secs",
            "full_circuit_exec_time_secs",
            "full_circuit_total_runtime_secs",
            "circuit_cutting_expval",
            "full_circuit_expval",
            "error_in_estimation",
            "scenario_label"
        ]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)
        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Check if the logger already has handlers to prevent duplicates
        if not logger.hasHandlers():
            # Create a console handler and set the log level
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create a formatter and add it to the console handler
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)

            # Add the console handler to the logger
            logger.addHandler(console_handler)

        self.logger = logger

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'metrics_file_writer'):
            self.metrics_file_writer.close()

    def pre_processing(self, circuit, observable, num_samples=10):
        """
        Preprocess the circuit by finding cuts and generating subexperiments.
        
        Args:
            circuit (QuantumCircuit): The quantum circuit to process
            observable (SparsePauliOp): The observable to measure
            num_samples (int): Number of samples to generate
            
        Returns:
            tuple: Contains subexperiments, coefficients, subobservables, 
                  original observable, and circuit
        """
        # Specify settings for the cut-finding optimizer
        optimization_settings = OptimizationParameters(seed=111)

        # Specify the size of the QPUs available
        device_constraints = DeviceConstraints(
            qubits_per_subcircuit=self.subcircuit_size
        )

        cut_circuit, metadata = find_cuts(
            circuit, optimization_settings, device_constraints
        )
        self.metadata = metadata

        self.logger.info(
            f"Full circuit size: {len(circuit.qubits)} \n"
            f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
            f'Sampling overhead of {metadata["sampling_overhead"]}.\n'
            f'Lowest cost solution found: {metadata["minimum_reached"]}.'
        )
        for cut in metadata["cuts"]:
            self.logger.info(f"{cut[0]} at circuit instruction index {cut[1]}")

        qc_w_ancilla = cut_wires(cut_circuit)
        observables_expanded = expand_observables(
            observable.paulis, circuit, qc_w_ancilla
        )

        partitioned_problem = partition_problem(
            circuit=qc_w_ancilla, observables=observables_expanded
        )
        subcircuits = partitioned_problem.subcircuits
        subobservables = partitioned_problem.subobservables
        self.logger.info(
            f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}"
        )

        subexperiments, coefficients = generate_cutting_experiments(
            circuits=subcircuits, observables=subobservables, num_samples=num_samples
        )
        self.logger.info(
            f"{sum(len(expts) for expts in subexperiments.values())} total subexperiments to run on backend."
        )

        return subexperiments, coefficients, subobservables, observable, circuit

    

    def run_circuit_cutting(self, circuit, observable, circuit_cutting_qiskit_options):
        """
        Executes the circuit cutting portion of the quantum simulation.
        
        Args:
            circuit (QuantumCircuit): The quantum circuit to be cut and executed
            observable (SparsePauliOp): The observable to measure
            pass_manager: The transpiler pass manager
            
        Returns:
            tuple: (final_expval, metrics) containing:
                - final_expval: The final expectation value
                - metrics: Dictionary containing timing and execution metrics
        """
        
        transpile_backend = AerSimulator(**circuit_cutting_qiskit_options["backend_options"])

        #transpile_backend = GenericBackendV2(num_qubits=circuit.num_qubits)
        pass_manager = generate_preset_pass_manager(
                optimization_level=0, backend=transpile_backend
            )
        # start time
        start_find_cuts = time.time()
        subexperiments, coefficients, subobservables, observable, circuit = (
            self.pre_processing(circuit, observable, self.num_samples)
        )
        end_find_cuts = time.time()

        transpile_start_time = time.time()
        self.logger.info(
            "*********************************** transpiling circuits ***********************************"
        )
        # Transpile the subexperiments to ISA circuits
        isa_subexperiments = {}
        for label, partition_subexpts in subexperiments.items():
            isa_subexperiments[label] = pass_manager.run(partition_subexpts, num_processes=1)
        self.logger.info(
            "*********************************** transpiling done ***************************************"
        )
        transpile_end_time = time.time()
        transpile_time_secs = transpile_end_time - transpile_start_time
        self.logger.info(f"Transpile time: {transpile_time_secs}")

        tasks = []
        sub_circuit_execution_time = time.time()
        resources = copy.copy(self.sub_circuit_task_resources)

        self.logger.info(
            f"********************** len of subexperiments {len(isa_subexperiments)}********************"
        )
        
        tasks = []
        active_tasks = []
        results_tuple = []
        number_of_tasks = 0

        # calculate the number of GPUs available
        if self.sub_circuit_task_resources["num_gpus"] > 0:
            num_slots = self.executor.cluster_config["config"]["gpus_per_node"]*self.executor.cluster_config["config"]["number_of_nodes"]
        else:
            num_slots = self.executor.cluster_config["config"]["number_of_nodes"]* self.executor.cluster_config["config"]["cores_per_node"]

        # Oversubscribe the number of slots
        num_slots = num_slots * 2

        self.logger.info(f"Number of slots available: {num_slots}")
        
        for label, subsystem_subexpts in isa_subexperiments.items():
            self.logger.info(
                f"*************** len of subsystem_subexpts {len(subsystem_subexpts)}**********"
            )
            # Create a queue of all experiments that need to be run
            experiment_queue = [(label, ss) for ss in subsystem_subexpts]
        
            while experiment_queue or active_tasks:
                # Submit new tasks if we have capacity and experiments waiting
                while len(active_tasks) < num_slots and experiment_queue:
                    label, ss = experiment_queue.pop(0)
                    task_future = self.executor.submit_task(
                        execute_sampler,
                        self.circuit_cutting_qiskit_options,
                        label,
                        [ss],
                        resources=resources,
                        shots=2**12,
                    )
                    active_tasks.append(task_future)
                    tasks.append(task_future)
                    number_of_tasks += 1
                    
                 # Check for completed tasks using ray.wait()
                if active_tasks:
                    ready_refs, remaining_refs = ray.wait(active_tasks, timeout=0.1)  # 100ms timeout
                
                # Process completed tasks
                for task_ref in ready_refs:
                    result = ray.get(task_ref)  # Get the result
                    results_tuple.append(result)
                    active_tasks.remove(task_ref)

            # if use_ray:
            # if use_ray:
            #     for ss in subsystem_subexpts:

            #         # if self.sub_circuit_task_resources["num_gpus"] > 0:                      
            #         #     while not self.check_gpu_availability():
            #         #         print("No GPU available, retrying...")
            #         #         time.sleep(1)
                       
            #         task_future = self.executor.submit_task(
            #             execute_sampler,
            #             self.circuit_cutting_qiskit_options,
            #             label,
            #             [ss],
            #             resources=resources,
            #             shots=2**12,
            #         )
                
            #         tasks.append(task_future)
            #         number_of_tasks = number_of_tasks + 1 
            # else:
            #     # sequential version
            #     for ss in subsystem_subexpts:
            #         result = execute_sampler(self.circuit_cutting_qiskit_options, label, [ss], shots=2**12)
            #         print(result)
            #         results_tuple.append(result)
            #         number_of_tasks = number_of_tasks + 1 

        # temporary fix for the parallel version
        # if use_ray:
        #     results_tuple = self.executor.get_results(tasks)

        sub_circuit_execution_end_time = time.time()
        subcircuit_exec_time_secs = (
            sub_circuit_execution_end_time - sub_circuit_execution_time
        )
        self.logger.info(f"Execution time for subcircuits: {subcircuit_exec_time_secs}")
        
        # Get all samplePubResults
        samplePubResults = collections.defaultdict(list)
        for result in results_tuple:
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
        reconstruct_subcircuit_expectations_time_secs = (
            reconstruction_end_time - reconstruction_start_time
        )

        total_runtime_secs = reconstruction_end_time - start_find_cuts

        self.logger.info(
            f"Execution time for reconstruction: {reconstruct_subcircuit_expectations_time_secs}"
        )

        final_expval = np.dot(reconstructed_expvals, observable.coeffs)
        self.logger.info(f"Reconstructed expectation value: {np.real(np.round(final_expval, 8))}")

        metrics =  {
            'find_cuts_time': end_find_cuts - start_find_cuts,
            'transpile_time': transpile_time_secs,
            'subcircuit_exec_time': subcircuit_exec_time_secs,
            'find_cuts_time': end_find_cuts - start_find_cuts,
            'reconstruction_time': reconstruct_subcircuit_expectations_time_secs,
            'total_runtime': total_runtime_secs,
            'number_of_tasks': number_of_tasks            
        }

        return final_expval, metrics

    def run_full_circuit_simulation(self, circuit, observable, full_circuit_qiskit_options):
        """
        Execute the full circuit simulation either via direct execution or MPI.
        
        Args:
            circuit (QuantumCircuit): The quantum circuit to simulate
            observable (SparsePauliOp): The observable to measure
            pass_manager: The transpiler pass manager
            full_circuit_qiskit_options (dict): Backend options for full circuit simulation
            
        Returns:
            tuple: (exact_expval, metrics) containing:
                - exact_expval (float): The expectation value from full circuit simulation
                - metrics (dict): Dictionary containing timing metrics
        """
        # from qiskit_aer import AerSimulator

        # transpiler_options = {"device":"CPU", "method":"statevector", "shots": 1024}
        # transpile_backend = AerSimulator(**transpiler_options)
        # pass_manager = generate_preset_pass_manager(
        #         optimization_level=1, backend=transpile_backend
        #     )
       
 
        transpile_backend = GenericBackendV2(num_qubits=circuit.num_qubits)
        pass_manager = generate_preset_pass_manager(0, transpile_backend)

        transpile_start = time.time()
        full_circuit_transpilation = pass_manager.run(circuit)
        transpile_end = time.time()
        transpile_time = transpile_end - transpile_start
        

        # backend = AerSimulator(**full_circuit_qiskit_options["backend_options"])
        self.logger.info(f"Execution time for full circuit transpilation: {transpile_time}")

        # Start timing the full circuit estimation
        estimator_start = time.time()

        # num_nodes attribute in task resource description in Ray
        ray_task_resources = {k: v for k, v in self.full_circuit_task_resources.items() if k != "num_nodes"}
        
        if "mpi" not in full_circuit_qiskit_options or full_circuit_qiskit_options["mpi"] == False:
            # Execute the circuit without
            # exact_expval = run_full_circuit(observable, full_circuit_qiskit_options, full_circuit_transpilation)
            
            # Submit task for non-MPI execution directly as a Ray / Dask task
            full_circuit_task = self.executor.submit_task(
                run_full_circuit,
                observable,
                full_circuit_qiskit_options,
                full_circuit_transpilation,
                resources=ray_task_resources
            )
            exact_expval = self.executor.get_results([full_circuit_task])
        else:
            
            # Serialize the circuit and observable to files
            # Get working directory from executor or use current directory as fallback
            working_dir = self.executor.cluster_config["config"]["working_directory"]
            
            # Create full paths using working directory
            hash_id = hex(hash(str(time.time())))[-5:]
            circuit_file = os.path.join(working_dir, f"full_circuit_{hash_id}.qpy")
            observable_file = os.path.join(working_dir, f"observable_{hash_id}.npy")
            backend_file = os.path.join(working_dir, f"backend_options_{hash_id}.json")

            # Write files to working directory
            with open(circuit_file, "wb") as f:
                qpy.dump(full_circuit_transpilation, f)

            np.save(observable_file, observable.to_list())
            
            with open(backend_file, "w") as f:
                json.dump(full_circuit_qiskit_options, f)

            num_nodes = self.full_circuit_task_resources.get("num_nodes", 1)  # Default to 1 if not specified
          
            # Submit task for MPI parallel execution via command line

            cmd = ["srun", "-N", str(num_nodes), 
                   f"--ntasks-per-node={self.full_circuit_task_resources['num_gpus']}", "--gpus-per-task=1" , "python", "-m", 
                   "mini_apps.quantum_simulation.motifs.circuit_cutting_motif",
                   observable_file, backend_file, circuit_file]
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            task = self.executor.submit_task(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                resources=ray_task_resources
            )
            result = self.executor.get_results([task])[0]
        
            if result.returncode != 0:
                raise RuntimeError(f"Command failed with error: {result.stderr}")
                       
            # Log the number of output lines
            output_lines = result.stdout.strip().split('\n')
            self.logger.info(f"Number of output lines: {len(output_lines)}")
       
            
            # result = subprocess.run(cmd, capture_output=True, text=True)
            exact_expval = float(output_lines[0])

            # delete the files
            os.remove(circuit_file)
            os.remove(observable_file)
            os.remove(backend_file)

        estimator_time = time.time() - estimator_start
        
        self.logger.info(f"Execution time for full circuit: {estimator_time}")
        self.logger.info(f"Exact expectation value: {np.round(exact_expval, 8)}")
        
        return exact_expval, {
            'transpile_time': transpile_time,
            'estimator_time': estimator_time,
            'total_time': transpile_time + estimator_time
        }

    def run(self):
        
        self.logger.info(f"Circuit Size: {self.base_qubits} Running Full Circuit Only: {self.full_circuit_only} Circuit Cutting Only: {self.circuit_cutting_only}" )

        # Configure backend and transpiler
        circuit_cutting_qiskit_options = DEFAULT_SIMULATOR_BACKEND_OPTIONS
        full_circuit_qiskit_options = DEFAULT_SIMULATOR_BACKEND_OPTIONS
        
        if self.full_circuit_qiskit_options is not None:
            full_circuit_qiskit_options = self.full_circuit_qiskit_options
        
        if self.circuit_cutting_qiskit_options is not None:
            circuit_cutting_qiskit_options = self.circuit_cutting_qiskit_options
        
        self.logger.info(f"Circuit Cutting Backend options: {circuit_cutting_qiskit_options}")
        self.logger.info(f"Full Circuit Backend options: {full_circuit_qiskit_options}")
        
        circuit, observable = self._generate_circuit_and_observable()

        if self.full_circuit_only == False: # Run circuit cutting experiments
            final_expval, metrics = self.run_circuit_cutting(circuit, observable, circuit_cutting_qiskit_options)
            
            # Store metrics for later use
            circuit_cutting_transpile_time_secs = metrics['transpile_time']
            circuit_cutting_exec_time_secs = metrics['subcircuit_exec_time']
            circuit_cutting_find_cuts_time_secs = metrics['find_cuts_time']
            circuit_cutting_reconstruct_subcircuit_expectations_time_secs = metrics['reconstruction_time']
            circuit_cutting_total_runtime_secs = metrics['total_runtime']
            number_of_tasks = metrics['number_of_tasks']

        if self.circuit_cutting_only == False: # Run full circuit simulation
            exact_expval, full_metrics = self.run_full_circuit_simulation(circuit, observable, full_circuit_qiskit_options)

            # Store full circuit metrics analogously
            full_circuit_transpile_time_secs = full_metrics['transpile_time']
            full_circuit_exec_time_sec = full_metrics['estimator_time']
            full_circuit_total_runtime_secs = full_metrics['total_time']


        # Calculate error in estimation between circuit cutting and full circuit simulation
        if self.full_circuit_only == False and self.circuit_cutting_only == False:
            error_in_estimation = np.real(np.round(final_expval - exact_expval, 8))
            self.logger.info(f"Error in estimation: {error_in_estimation}")
            self.logger.info(
                f"Relative error in estimation: {np.real(np.round((final_expval-exact_expval) / exact_expval, 8))}"
            )
        
        # Write metrics to file
        self.metrics_file_writer.write(
            [
                getattr(self, "experiment_start_time", None),
                getattr(self, "subcircuit_size", None),
                getattr(self, "base_qubits", None),
                getattr(self, "observables", None),
                getattr(self, "scale_factor", None),
                getattr(self, "num_samples", None),
                number_of_tasks if 'number_of_tasks' in locals() else None,
                str(self.metadata) if hasattr(self, "metadata") else None,
                str(self.executor.cluster_config) if hasattr(self.executor, "cluster_config") else None,
                str(self.full_circuit_qiskit_options) if hasattr(self, "full_circuit_qiskit_options") else None,
                str(self.circuit_cutting_qiskit_options) if hasattr(self, "circuit_cutting_qiskit_options") else None,
                str(self.sub_circuit_task_resources) if hasattr(self, "sub_circuit_task_resources") else None,
                str(self.full_circuit_task_resources) if hasattr(self, "full_circuit_task_resources") else None,
                circuit_cutting_find_cuts_time_secs if 'circuit_cutting_find_cuts_time_secs' in locals() else None,
                circuit_cutting_transpile_time_secs if 'circuit_cutting_transpile_time_secs' in locals() else None,
                circuit_cutting_exec_time_secs if 'circuit_cutting_exec_time_secs' in locals() else None,
                circuit_cutting_reconstruct_subcircuit_expectations_time_secs if 'circuit_cutting_reconstruct_subcircuit_expectations_time_secs' in 
                locals() else None,                
                circuit_cutting_total_runtime_secs if 'circuit_cutting_total_runtime_secs' in locals() else None,
                full_circuit_transpile_time_secs if 'full_circuit_transpile_time_secs' in locals() else None,
                full_circuit_exec_time_sec if 'full_circuit_exec_time_sec' in locals() else None,
                full_circuit_total_runtime_secs if 'full_circuit_total_runtime_secs' in locals() else None,
                final_expval if 'final_expval' in locals() else None,
                exact_expval if 'exact_expval' in locals() else None,
                float(error_in_estimation) if 'error_in_estimation' in locals() else None,
                self.scenario_label
            ]
        )

        self.metrics_file_writer.close()


    def _generate_circuit_and_observable(self):
        """
        Generates a quantum circuit and an observable.

        This method creates a random quantum circuit using the EfficientSU2 ansatz with a specified
        number of qubits and entanglement pattern. The circuit parameters are assigned a fixed value.
        It also constructs an observable by scaling the provided observables.

        Returns:
            Tuple[QuantumCircuit, SparsePauliOp]: A tuple containing the generated quantum circuit and the observable.
        """
        # Generate standard circuit comprising of 1 single qubit rotation and 1 entangling  gates between all qubits for each layer
        circuit = EfficientSU2(self.base_qubits * self.scale_factor, entanglement="linear", reps=2).decompose()
        circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)

        observable = SparsePauliOp([o * self.scale_factor for o in self.observables])

        return circuit, observable


    def write_metrics(self, metrics_data):
        """
        Safely write metrics data with validation.
        
        Args:
            metrics_data (list): List of metric values to write
        """
        # Validate all required metrics are present
        if len(metrics_data) != len(self.metrics_file_writer.header):
            raise ValueError("Metrics data length does not match header length")
        
        # Convert None values to appropriate format
        formatted_data = ['' if x is None else x for x in metrics_data]
        
        self.metrics_file_writer.write(formatted_data)





if __name__ == '__main__':
    """
    Run the full circuit simulation from the command line 
    Used for testing the full circuit simulation with distributed state vector simulation based on MPI
    """
    fire.Fire(cli_run_full_circuit)