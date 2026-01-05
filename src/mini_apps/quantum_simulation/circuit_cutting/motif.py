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
from engine.base.base_motif import Motif
from mini_apps.quantum_simulation.circuit_execution.motifs.qiskit_benchmark import generate_data


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
# Helper Functions

def log_subexperiment_characteristics(logger, task_id, label, circuit, stage="SUBMIT"):
    """
    Log circuit characteristics for straggler analysis.

    Args:
        logger: Logger instance
        task_id: Task identifier
        label: Subsystem label (e.g., 'A', 'B', 'C')
        circuit: Qiskit QuantumCircuit object
        stage: Logging stage ('SUBMIT' or 'EXEC')
    """
    from qiskit.converters import circuit_to_dag

    two_q_gates = len(circuit_to_dag(circuit).two_qubit_ops())
    gate_counts = circuit.count_ops()

    logger.info(
        f"[SUBEXPT_{stage}] "
        f"task_id={task_id}, "
        f"label={label}, "
        f"qubits={circuit.num_qubits}, "
        f"depth={circuit.depth()}, "
        f"total_gates={circuit.size()}, "
        f"two_q_gates={two_q_gates}, "
        f"cx_gates={gate_counts.get('cx', 0)}, "
        f"gate_types={dict(gate_counts)}"
    )


##################################################################################################
# Called from distributed executor, e.g., Ray or MPI

# Circuit Cutting Simulation
def execute_sampler(backend_options, label, subsystem_subexpts, 
                    shots, num_threads=1):
    # Add error handling
    try: 
        from qiskit_aer import AerSimulator
        from qiskit.converters import circuit_to_dag

        task_start_time = time.time()  # Track total task time

        # Log circuit characteristics when execution starts
        circuit = subsystem_subexpts[0]
        two_q_gates = len(circuit_to_dag(circuit).two_qubit_ops())

        print(f"[SUBEXPT_EXEC_START] "
              f"label={label}, "
              f"threads={num_threads}, "
              f"qubits={circuit.num_qubits}, "
              f"depth={circuit.depth()}, "
              f"total_gates={circuit.size()}, "
              f"two_q_gates={two_q_gates}, "
              f"device={backend_options['backend_options'].get('device', 'N/A')}, "
              f"shots={shots}")

        submit_start = time.time()
        # Configure Aer to use multiple threads
        backend_opts = backend_options["backend_options"].copy()
        backend_opts["max_parallel_threads"] = num_threads
        backend = AerSimulator(**backend_opts)

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

            task_end_time = time.time()
            total_task_time = task_end_time - task_start_time

            print(
                f"Job {label} completed with job id {job.job_id()}, "
                f"submit_time: {submit_end-submit_start:.3f}s, "
                f"execution_time: {result_end - result_start:.3f}s, "
                f"total_task_time: {total_task_time:.3f}s, "
                f"type: {type(new_result)}"
            )
            return (label, new_result, total_task_time)
    except Exception as e:
        logging.error(f"Error executing sampler: {str(e)}")
        return None  # Return None explicitly when there's an error


# Full Circuit Simulation
def run_full_circuit(observable, backend_options, full_circuit, num_threads=1):
    try:
        from qiskit_aer.primitives import EstimatorV2
        from qiskit_aer import AerSimulator

        # Configure Aer to use multiple threads
        backend_opts = backend_options["backend_options"].copy()
        backend_opts["max_parallel_threads"] = num_threads

        # Create simulator
        simulator = AerSimulator(**backend_opts)

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
    from mini_apps.quantum_simulation.circuit_cutting.motif import run_full_circuit
    result = run_full_circuit(observable, backend_options, full_circuit)
    
    # Convert numpy types to Python native types for JSON serialization
    if isinstance(result, np.ndarray):
        result = result.tolist()

    # print(f"{result}")    
    return result



################################################################################

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
        self.max_parallel_tasks = None  # Optional limit from QDreamer
        self.gpu_mode = False  # Track if GPU acceleration is enabled
        self.gpu_fraction = None  # Track GPU fraction per task
        self.gpu_type = None  # Track GPU type (e.g., 'B200', 'H100', 'A100')
        self.custom_circuit = None  # Optional: pre-generated circuit to use instead of generating one
        self.default_circuit_depth = 2  # Default circuit depth (reps) for EfficientSU2
        # Parallelization configuration tracking (for CSV output)
        self.num_nodes = None
        self.cores_per_node = None
        self.gpus_per_node = None
        self.result_file = None  # Optional: CSV file to write results
        self.scenario_label = None  # Optional: label for the scenario

    def set_circuit(self, circuit):
        """
        Set a pre-generated circuit to use instead of generating a new one.
        This ensures the same circuit is used for both QDreamer optimization and actual execution.
        
        Args:
            circuit: QuantumCircuit object to use
            
        Returns:
            CircuitCuttingBuilder: self for method chaining
        """
        self.custom_circuit = circuit
        return self

    def set_circuit_depth(self, circuit_depth):
        """
        Set the circuit depth (reps) for EfficientSU2 circuit generation.
        Only used when no custom circuit is provided.
        
        Args:
            circuit_depth: Number of repetitions/layers for EfficientSU2 circuit (default: 2)
            
        Returns:
            CircuitCuttingBuilder: self for method chaining
        """
        self.default_circuit_depth = circuit_depth
        return self

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

    def set_max_parallel_tasks(self, max_parallel_tasks):
        """Set maximum number of parallel tasks (e.g., from QDreamer optimization)"""
        self.max_parallel_tasks = max_parallel_tasks
        return self

    def set_qdreamer_allocation(self, allocation):
        """Optional: Set QDreamer allocation for prediction tracking in CSV output"""
        self.qdreamer_allocation = allocation
        return self

    def set_gpu_mode(self, gpu_mode):
        """Set GPU mode flag for tracking in CSV output"""
        self.gpu_mode = gpu_mode
        return self

    def set_gpu_fraction(self, gpu_fraction):
        """Set GPU fraction per task for tracking in CSV output"""
        self.gpu_fraction = gpu_fraction
        return self

    def set_gpu_type(self, gpu_type):
        """Set GPU type (e.g., 'B200', 'H100', 'A100') for tracking in CSV output"""
        self.gpu_type = gpu_type
        return self

    def set_num_nodes(self, num_nodes):
        """Set number of nodes in the cluster for CSV tracking"""
        self.num_nodes = num_nodes
        return self

    def set_cores_per_node(self, cores_per_node):
        """Set number of CPU cores per node for CSV tracking"""
        self.cores_per_node = cores_per_node
        return self

    def set_gpus_per_node(self, gpus_per_node):
        """Set number of GPUs per node for CSV tracking"""
        self.gpus_per_node = gpus_per_node
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
            self.scenario_label,
            self.max_parallel_tasks,
            getattr(self, 'qdreamer_allocation', None),  # Optional: None if not set
            self.gpu_mode,
            self.gpu_fraction,
            self.gpu_type,
            getattr(self, 'custom_circuit', None),  # Pass custom circuit if set
            self.num_nodes,
            self.cores_per_node,
            self.gpus_per_node,
            self.default_circuit_depth
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
        scenario_label,
        max_parallel_tasks=None,
        qdreamer_allocation=None,
        gpu_mode=False,
        gpu_fraction=None,
        gpu_type=None,
        custom_circuit=None,
        num_nodes=None,
        cores_per_node=None,
        gpus_per_node=None,
        default_circuit_depth=2
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
        self.max_parallel_tasks = max_parallel_tasks  # Optional limit from QDreamer
        self.qdreamer_allocation = qdreamer_allocation  # Optional QDreamer predictions
        self.gpu_mode = gpu_mode  # Track if GPU acceleration is enabled
        self.gpu_fraction = gpu_fraction  # Track GPU fraction per task
        self.gpu_type = gpu_type  # Track GPU type (e.g., 'B200', 'H100', 'A100')
        self.custom_circuit = custom_circuit  # Optional: pre-generated circuit to use (ensures same as QDreamer)
        self.default_circuit_depth = default_circuit_depth  # Circuit depth (reps) for default EfficientSU2
        # Parallelization configuration tracking
        self.num_nodes = num_nodes
        self.cores_per_node = cores_per_node
        self.gpus_per_node = gpus_per_node
        self.metadata = None

        header = [
            "experiment_start_time",
            "subcircuit_size",
            "base_qubits",
            "circuit_type",
            "circuit_depth",
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
            "scenario_label",
            # QDreamer prediction columns (optional, empty if QDreamer not used)
            "num_cuts",
            "sampling_overhead",
            "predicted_speedup",
            "actual_speedup",
            "speedup_error",
            "prediction_accuracy_pct",
            # GPU configuration tracking
            "gpu_mode",
            "gpu_fraction",
            "gpu_type",
            "num_slots",
            # Cut type tracking
            "num_gate_cuts",
            "num_wire_cuts",
            # Parallelization configuration
            "num_nodes",
            "cores_per_node",
            "gpus_per_node",
            # Task runtime tracking
            "task_runtimes"
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
        # CRITICAL: Use the same seed as QDreamer if available, otherwise default to 111
        if hasattr(self, 'qdreamer_allocation') and self.qdreamer_allocation is not None:
            # Try to get seed from QDreamer allocation metadata if available
            seed = self.qdreamer_allocation.metadata.get('optimization_seed', 111) if hasattr(self.qdreamer_allocation, 'metadata') else 111
        else:
            seed = 111
        optimization_settings = OptimizationParameters(seed=seed)

        # Specify the size of the QPUs available
        device_constraints = DeviceConstraints(
            qubits_per_subcircuit=self.subcircuit_size
        )

        cut_circuit, metadata = find_cuts(
            circuit, optimization_settings, device_constraints
        )
        self.metadata = metadata
        
        # Parse and count gate cuts vs wire cuts
        self.num_gate_cuts = 0
        self.num_wire_cuts = 0
        if metadata and "cuts" in metadata:
            for cut in metadata["cuts"]:
                cut_type = cut[0] if isinstance(cut, tuple) else str(cut)
                if "Gate" in cut_type or "gate" in cut_type:
                    self.num_gate_cuts += 1
                elif "Wire" in cut_type or "wire" in cut_type:
                    self.num_wire_cuts += 1

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

        # Remove None values and non-Ray options from resources dict
        # num_threads is a custom field for execute_sampler, not a Ray option
        resources = {k: v for k, v in resources.items()
                     if v is not None and k not in ["num_threads"]}

        self.logger.info(f"[TASK RESOURCES] Filtered resources dict passed to Ray: {resources}")
        self.logger.info(
            f"********************** len of subexperiments {len(isa_subexperiments)}********************"
        )
        
        tasks = []
        active_tasks = []
        results_tuple = []
        number_of_tasks = 0

        # calculate the number of GPUs available
        # Safely access cluster config with fallbacks
        try:
            config = self.executor.cluster_config.get("config", {})
            gpus_per_node = config.get("gpus_per_node", 0)
            cores_per_node = config.get("cores_per_node", 1)
            number_of_nodes = config.get("number_of_nodes", 1)

            # Log the configuration for debugging
            self.logger.info(f"[DEBUG] Task resource configuration:")
            self.logger.info(f"  sub_circuit_task_resources: {self.sub_circuit_task_resources}")
            self.logger.info(f"  gpus_per_node: {gpus_per_node}")
            self.logger.info(f"  cores_per_node: {cores_per_node}")
            self.logger.info(f"  number_of_nodes: {number_of_nodes}")

            # Check if tasks are actually requesting GPUs AND GPUs are available in cluster
            if self.sub_circuit_task_resources.get("num_gpus", 0) > 0 and gpus_per_node > 0:
                num_slots = gpus_per_node * number_of_nodes
                self.logger.info(f"  Using GPU-based parallelism: {num_slots} GPU slots")
            else:
                num_slots = number_of_nodes * cores_per_node
                self.logger.info(f"  Using CPU-based parallelism: {num_slots} CPU slots")
        except (AttributeError, KeyError, TypeError) as e:
            self.logger.warning(f"Could not determine num_slots from executor config: {e}")
            # Fallback: use reasonable default based on requested resources
            if self.sub_circuit_task_resources.get("num_gpus", 0) > 0:
                num_slots = 1  # At least 1 GPU slot
                self.logger.warning(f"  Fallback: Using 1 GPU slot")
            else:
                num_slots = max(1, self.sub_circuit_task_resources.get("num_cpus", 1))
                self.logger.warning(f"  Fallback: Using {num_slots} CPU slot(s)")

        # Don't oversubscribe to avoid OOM - keep 1:1 with GPUs
        # num_slots = num_slots * 2

        # Validate num_slots
        if num_slots == 0:
            try:
                error_msg = (
                    f"Invalid configuration: num_slots=0. "
                    f"Config: gpus_per_node={config.get('gpus_per_node', 'N/A')}, "
                    f"cores_per_node={config.get('cores_per_node', 'N/A')}, "
                    f"number_of_nodes={config.get('number_of_nodes', 'N/A')}"
                )
            except:
                error_msg = f"Invalid configuration: num_slots=0 (config unavailable)"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Apply QDreamer's max_parallel_tasks recommendation if provided
        if self.max_parallel_tasks is not None:
            original_num_slots = num_slots
            num_slots = min(num_slots, self.max_parallel_tasks)
            if original_num_slots != num_slots:
                self.logger.info(
                    f"[QDREAMER] Limiting parallelism from {original_num_slots} to {num_slots} "
                    f"based on QDreamer optimization"
                )

        self.logger.info(f"[FINAL] Number of slots available for parallel execution: {num_slots}")
        self.logger.info(f"[FINAL] Each task will use: {self.sub_circuit_task_resources}")
        
        # Store num_slots for CSV output
        self.num_slots = num_slots

        # FLATTENED APPROACH: Combine all experiments from all subsystems into a single queue
        # This allows full parallelization across ALL experiments (not just within each subsystem)
        self.logger.info("=" * 80)
        self.logger.info("FLATTENING experiments from all subsystems for parallel execution")
        self.logger.info(f"Number of subsystems: {len(isa_subexperiments)}")

        # RAY DIAGNOSTICS: Check Ray's view of available resources
        try:
            import ray
            ray_resources = ray.available_resources()
            self.logger.info(f"[RAY DIAGNOSTICS] Available resources from Ray: {ray_resources}")
            self.logger.info(f"[RAY DIAGNOSTICS] Cluster resources: {ray.cluster_resources()}")
        except Exception as e:
            self.logger.warning(f"Could not get Ray diagnostics: {e}")

        all_experiments = []
        for label, subsystem_subexpts in isa_subexperiments.items():
            num_expts = len(subsystem_subexpts)
            self.logger.info(f"  Subsystem '{label}': {num_expts} subexperiments")
            for ss in subsystem_subexpts:
                all_experiments.append((label, ss))

        total_experiments = len(all_experiments)
        self.logger.info(f"Total experiments to execute: {total_experiments}")
        self.logger.info(f"Max concurrent tasks: {num_slots}")
        
        # STRAGGLER MITIGATION: Sort experiments by complexity (most complex first)
        # This "Longest Job First" strategy minimizes makespan by scheduling
        # expensive tasks early, preventing stragglers from accumulating at the end
        self.logger.info("=" * 80)
        self.logger.info("STRAGGLER MITIGATION: Sorting experiments by complexity")
        
        # Calculate complexity metric for each experiment (depth × gates)
        experiments_with_complexity = []
        for label, circuit in all_experiments:
            complexity = circuit.depth() * circuit.size()  # depth × total gates
            experiments_with_complexity.append((label, circuit, complexity))
        
        # Sort by complexity (highest first)
        experiments_with_complexity.sort(key=lambda x: x[2], reverse=True)
        
        # Log complexity distribution
        complexities = [x[2] for x in experiments_with_complexity]
        self.logger.info(f"  Complexity range: {min(complexities):.0f} - {max(complexities):.0f}")
        self.logger.info(f"  Complexity mean: {np.mean(complexities):.0f}")
        self.logger.info(f"  Complexity median: {np.median(complexities):.0f}")
        self.logger.info(f"  Most complex task scheduled first (complexity={complexities[0]:.0f})")
        self.logger.info(f"  Least complex task scheduled last (complexity={complexities[-1]:.0f})")
        
        # Create queue with sorted experiments (remove complexity metric)
        experiment_queue = [(label, circuit) for label, circuit, _ in experiments_with_complexity]
        
        # Log sample of scheduled order for verification
        self.logger.info(f"  First 5 tasks (most complex):")
        for i in range(min(5, len(experiments_with_complexity))):
            label, circuit, complexity = experiments_with_complexity[i]
            self.logger.info(f"    Task {i}: depth={circuit.depth()}, gates={circuit.size()}, complexity={complexity:.0f}")
        
        if len(experiments_with_complexity) > 10:
            self.logger.info(f"  Last 5 tasks (least complex):")
            for i in range(max(0, len(experiments_with_complexity)-5), len(experiments_with_complexity)):
                label, circuit, complexity = experiments_with_complexity[i]
                self.logger.info(f"    Task {i}: depth={circuit.depth()}, gates={circuit.size()}, complexity={complexity:.0f}")
        
        self.logger.info("=" * 80)
        iteration_count = 0
        last_log_time = time.time()

        while experiment_queue or active_tasks:
            iteration_count += 1

            # Submit new tasks if we have capacity and experiments waiting
            while len(active_tasks) < num_slots and experiment_queue:
                label, ss = experiment_queue.pop(0)

                # Log circuit characteristics at submission for straggler analysis
                log_subexperiment_characteristics(
                    self.logger,
                    number_of_tasks + 1,  # task_id
                    label,
                    ss,  # circuit
                    stage="SUBMIT"
                )

                # Log first few task submissions for debugging
                if number_of_tasks < 3:
                    self.logger.info(
                        f"[TASK SUBMIT] Task {number_of_tasks + 1}: "
                        f"label={label}, resources={resources}, "
                        f"num_slots={num_slots}, queue_remaining={len(experiment_queue)}"
                    )

                # Get number of threads from task resources
                num_threads = self.sub_circuit_task_resources.get('num_threads', 1)

                task_future = self.executor.submit_task(
                    execute_sampler,
                    self.circuit_cutting_qiskit_options,
                    label,
                    [ss],
                    2**12,  # shots
                    num_threads,  # num_threads
                    resources=resources,
                )
                active_tasks.append(task_future)
                tasks.append(task_future)
                number_of_tasks += 1

            # Log progress periodically (every 2 seconds)
            current_time = time.time()
            if current_time - last_log_time >= 2.0:
                completed = total_experiments - len(experiment_queue) - len(active_tasks)
                self.logger.info(
                    f"[PROGRESS] Iter {iteration_count}: "
                    f"Completed: {completed}/{total_experiments}, "
                    f"Active: {len(active_tasks)}, "
                    f"Queued: {len(experiment_queue)}"
                )
                last_log_time = current_time

            # Check for completed tasks using ray.wait()
            ready_refs = []
            if active_tasks:
                ready_refs, remaining_refs = ray.wait(active_tasks, timeout=0.1)  # 100ms timeout
                # Update active_tasks to only include remaining tasks
                active_tasks = list(remaining_refs)

            # Process completed tasks
            for task_ref in ready_refs:
                result = ray.get(task_ref)  # Get the result
                results_tuple.append(result)

                # Log task completion with timing for straggler analysis
                if result and len(result) >= 3:
                    label, primitive_result, task_time = result[:3]
                    self.logger.info(
                        f"[SUBEXPT_COMPLETE] label={label}, task_time={task_time:.3f}s"
                    )

        self.logger.info("=" * 80)
        self.logger.info(f"All {total_experiments} experiments completed successfully!")
        self.logger.info("=" * 80)

        sub_circuit_execution_end_time = time.time()
        subcircuit_exec_time_secs = (
            sub_circuit_execution_end_time - sub_circuit_execution_time
        )
        self.logger.info(f"Execution time for subcircuits: {subcircuit_exec_time_secs}")

        # Collect task timing statistics
        task_times = []
        samplePubResults = collections.defaultdict(list)

        for result in results_tuple:
            if result is None:
                self.logger.warning("Skipping None result from failed task")
                continue

            # Unpack result tuple: (label, primitive_result, task_time)
            if len(result) == 3:
                label, primitive_result, task_time = result
                task_times.append(task_time)
            else:
                # Fallback for old format (label, primitive_result)
                label, primitive_result = result[0], result[1]

            self.logger.info(f"Result: {label}")
            samplePubResults[label].extend(primitive_result._pub_results)

        # Log task time statistics
        if task_times:
            self.logger.info("=" * 80)
            self.logger.info("TASK EXECUTION TIME STATISTICS")
            self.logger.info("=" * 80)
            self.logger.info(f"  Number of tasks: {len(task_times)}")
            self.logger.info(f"  Min task time: {min(task_times):.3f}s")
            self.logger.info(f"  Max task time: {max(task_times):.3f}s")
            self.logger.info(f"  Mean task time: {np.mean(task_times):.3f}s")
            self.logger.info(f"  Median task time: {np.median(task_times):.3f}s")
            self.logger.info(f"  Std dev: {np.std(task_times):.3f}s")
            self.logger.info(f"  Coefficient of variation: {np.std(task_times)/np.mean(task_times)*100:.1f}%")
            self.logger.info(f"  Range (max-min): {max(task_times) - min(task_times):.3f}s")
            self.logger.info("=" * 80)

        # Check if we have any valid results
        if not samplePubResults:
            self.logger.error("No valid results obtained from circuit cutting execution")
            return None, {"error": "No valid results"}

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
        final_expval = float(np.real(final_expval))  # Convert to float 
        self.logger.info(f"Reconstructed expectation value: {np.real(np.round(final_expval, 8))}")

        metrics =  {
            'find_cuts_time': end_find_cuts - start_find_cuts,
            'transpile_time': transpile_time_secs,
            'subcircuit_exec_time': subcircuit_exec_time_secs,
            'reconstruction_time': reconstruct_subcircuit_expectations_time_secs,
            'total_runtime': total_runtime_secs,
            'number_of_tasks': number_of_tasks,
            'task_times': task_times
        }

        # Force garbage collection and wait for Ray to clean up resources
        import gc
        gc.collect()

        # Give Ray time to release GPU resources from completed tasks
        self.logger.info("Waiting for GPU resources to be released...")
        time.sleep(2)

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
        # For MPI tasks, remove memory constraint since srun manages resources
        # Also remove mpi_ranks and num_threads as they are custom fields not recognized by Ray
        ray_task_resources = {k: v for k, v in self.full_circuit_task_resources.items()
                             if k not in ["num_nodes", "memory", "mpi_ranks", "num_threads"]}

        # Get number of threads from task resources
        num_threads = self.full_circuit_task_resources.get('num_threads', 1)

        if "mpi" not in full_circuit_qiskit_options or full_circuit_qiskit_options["mpi"] == False:
            # Execute the circuit without
            # exact_expval = run_full_circuit(observable, full_circuit_qiskit_options, full_circuit_transpilation)

            # Submit task for non-MPI execution directly as a Ray / Dask task
            full_circuit_task = self.executor.submit_task(
                run_full_circuit,
                observable,
                full_circuit_qiskit_options,
                full_circuit_transpilation,
                num_threads,  # num_threads
                resources=ray_task_resources
            )
            result = self.executor.get_results([full_circuit_task])
            exact_expval = result[0] if isinstance(result, list) else result

            # Check if the result is an error string
            if isinstance(exact_expval, str):
                raise RuntimeError(f"Full circuit simulation failed: {exact_expval}")
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

            # Get number of MPI ranks per node
            # Prefer mpi_ranks field if available, otherwise fall back to num_gpus
            ntasks_per_node = self.full_circuit_task_resources.get('mpi_ranks')
            if ntasks_per_node is None or ntasks_per_node == 0:
                ntasks_per_node = self.full_circuit_task_resources.get('num_gpus', 1)
                if ntasks_per_node == 0:
                    ntasks_per_node = 1  # Final fallback if num_gpus was 0

            cmd = ["srun", "-N", str(num_nodes),
                   f"--ntasks-per-node={ntasks_per_node}", "--gpus-per-task=1", "python", "-m",
                   "mini_apps.quantum_simulation.circuit_cutting.motif",
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

            # Check if circuit cutting was successful
            if final_expval is None or 'error' in metrics:
                self.logger.error("Circuit cutting failed, cannot proceed")
                return

            # Store metrics for later use
            circuit_cutting_transpile_time_secs = metrics['transpile_time']
            circuit_cutting_exec_time_secs = metrics['subcircuit_exec_time']
            circuit_cutting_find_cuts_time_secs = metrics['find_cuts_time']
            circuit_cutting_reconstruct_subcircuit_expectations_time_secs = metrics['reconstruction_time']
            circuit_cutting_total_runtime_secs = metrics['total_runtime']
            number_of_tasks = metrics['number_of_tasks']
            task_times = metrics['task_times']

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

        # Calculate QDreamer prediction metrics (if QDreamer was used)
        actual_speedup = None
        speedup_error = None
        prediction_accuracy = None

        if self.qdreamer_allocation and self.full_circuit_only == False and self.circuit_cutting_only == False:
            # Calculate actual speedup from execution times
            if 'circuit_cutting_total_runtime_secs' in locals() and 'full_circuit_total_runtime_secs' in locals():
                if circuit_cutting_total_runtime_secs > 0:
                    actual_speedup = full_circuit_total_runtime_secs / circuit_cutting_total_runtime_secs
                    predicted_speedup = self.qdreamer_allocation.speedup_factor
                    speedup_error = abs(actual_speedup - predicted_speedup)
                    if max(predicted_speedup, actual_speedup) > 0:
                        prediction_accuracy = (min(predicted_speedup, actual_speedup) / max(predicted_speedup, actual_speedup) * 100)

        # Compute qdreamer_num_cuts for metrics (None if QDreamer not used)
        qdreamer_num_cuts = self.qdreamer_allocation.num_cuts if self.qdreamer_allocation else None

        # Write metrics to file
        self.metrics_file_writer.write(
            [
                getattr(self, "experiment_start_time", None),
                getattr(self, "subcircuit_size", None),
                getattr(self, "base_qubits", None),
                getattr(self, "circuit_type", None),
                getattr(self, "circuit_depth", None),
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
                self.scenario_label,
                # QDreamer prediction columns (None if QDreamer not used)
                qdreamer_num_cuts,
                self.qdreamer_allocation.sampling_overhead if self.qdreamer_allocation else None,
                self.qdreamer_allocation.speedup_factor if self.qdreamer_allocation else None,
                actual_speedup,
                speedup_error,
                prediction_accuracy,
                # GPU configuration
                self.gpu_mode if hasattr(self, 'gpu_mode') else None,
                self.gpu_fraction if hasattr(self, 'gpu_fraction') else None,
                self.gpu_type if hasattr(self, 'gpu_type') else None,
                self.num_slots if hasattr(self, 'num_slots') else None,
                # Cut type counts
                self.num_gate_cuts if hasattr(self, 'num_gate_cuts') else None,
                self.num_wire_cuts if hasattr(self, 'num_wire_cuts') else None,
                # Parallelization configuration
                self.num_nodes if hasattr(self, 'num_nodes') else None,
                self.cores_per_node if hasattr(self, 'cores_per_node') else None,
                self.gpus_per_node if hasattr(self, 'gpus_per_node') else None,
                # Task runtimes as JSON array
                json.dumps(task_times) if 'task_times' in locals() else None
            ]
        )

        self.metrics_file_writer.close()


    def _generate_circuit_and_observable(self):
        """
        Generates a quantum circuit and an observable.

        This method creates a random quantum circuit using the EfficientSU2 ansatz with a specified
        number of qubits and entanglement pattern. The circuit parameters are assigned a fixed value.
        It also constructs an observable by scaling the provided observables.
        
        If a custom circuit was set via set_circuit(), it uses that instead of generating a new one.

        Returns:
            Tuple[QuantumCircuit, SparsePauliOp]: A tuple containing the generated quantum circuit and the observable.
        """
        # Use custom circuit if provided (ensures same circuit as QDreamer optimization)
        if hasattr(self, 'custom_circuit') and self.custom_circuit is not None:
            circuit = self.custom_circuit
            # Check if the custom circuit is an EfficientSU2 circuit (or derived from one)
            if isinstance(circuit, EfficientSU2):
                self.circuit_type = "EfficientSU2"
                self.circuit_depth = circuit.reps
                self.logger.info(f"Using pre-generated EfficientSU2 circuit ({circuit.num_qubits}q, reps={circuit.reps}, id={id(circuit)}) - matches QDreamer optimization")
            elif hasattr(circuit, '_efficientsu2_reps'):
                # Support for decomposed EfficientSU2 circuits that store reps as metadata
                self.circuit_type = "EfficientSU2"
                self.circuit_depth = circuit._efficientsu2_reps
                self.logger.info(f"Using pre-generated decomposed EfficientSU2 circuit ({circuit.num_qubits}q, reps={circuit._efficientsu2_reps}, id={id(circuit)}) - matches QDreamer optimization")
            else:
                self.circuit_type = "Custom"
                self.circuit_depth = circuit.depth()
                self.logger.info(f"Using pre-generated custom circuit ({circuit.num_qubits}q, {len(circuit)} gates, id={id(circuit)}) - matches QDreamer optimization")
        else:
            # Generate standard circuit comprising of 1 single qubit rotation and 1 entangling  gates between all qubits for each layer
            reps = self.default_circuit_depth
            circuit = EfficientSU2(self.base_qubits * self.scale_factor, entanglement="linear", reps=reps).decompose()
            circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
            self.circuit_type = "EfficientSU2"
            self.circuit_depth = reps

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