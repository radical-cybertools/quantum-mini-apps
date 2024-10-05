import os
import time


from qiskit import transpile
from qiskit_aer.primitives import Estimator as AirEstimator
from qiskit_ibm_runtime import EstimatorV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import Pauli
from qiskit_ionq import IonQProvider



from engine.metrics.csv_writer import MetricsFileWriter
from mini_apps.quantum_simulation.motifs.base_motif import Motif
from mini_apps.quantum_simulation.motifs.qiskit_benchmark import generate_data
import datetime
# from qiskit_rigetti import RigettiQCSProvider
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def run_circuit(circ_obs, qiskit_backend_options, simulator):
    transpiled_circuit = circ_obs[0]
    observable = circ_obs[1]
    estimator_result = None
    if simulator in ["ionq_simulator", "ionq_qpu"]: 
        key = qiskit_backend_options.get("api_key", None)
        if key is not None:
            key = os.environ["IONQ_API_KEY"]
        if key is None:
            ionq_provider = IonQProvider()
        else:
            ionq_provider = IonQProvider(token=key)
        backend = ionq_provider.get_backend(simulator)
        transpiled_circuit = transpile(circ_obs[0], backend=backend, optimization_level=3)   
        estimator_result = backend.run(transpiled_circuit).result()        
    elif simulator in ["aer_simulator"]:
        estimator = AirEstimator(backend_options=qiskit_backend_options)
        estimator_result = estimator.run(transpiled_circuit, Pauli(observable)).result()   
    else:
        service = QiskitRuntimeService()
        backend = service.backend(simulator)
        estimator_result = EstimatorV2(mode=backend).run([(transpiled_circuit, observable)]).result()        
    print(estimator_result)
    return estimator_result


class CircuitExecutionBuilder:
    def __init__(self):
        self.depth_of_recursion = 1
        self.num_qubits = 10
        self.n_entries = 10
        self.circuit_depth = 1
        self.size_of_observable = 1
        self.qiskit_backend_options = {"method": "statevector"}
        self.result_dir = os.environ['HOME']
        # create a date-time based file name
        self.current_datetime = datetime.datetime.now()
        self.file_name = f"ce_result_{self.current_datetime.strftime('%Y-%m-%dT%H:%M:%S')}.csv"
        self.result_file = os.path.join(self.result_dir, self.file_name)
        self.cluster_info = None  
        self.pilot = None  
        self.simulator = "aer_simulator"

    def set_depth_of_recursion(self, depth_of_recursion):
        self.depth_of_recursion = depth_of_recursion
        return self

    def set_num_qubits(self, num_qubits):
        self.num_qubits = num_qubits
        return self

    def set_n_entries(self, n_entries):
        self.n_entries = n_entries
        return self

    def set_circuit_depth(self, circuit_depth):
        self.circuit_depth = circuit_depth
        return self

    def set_size_of_observable(self, size_of_observable):
        self.size_of_observable = size_of_observable
        return self

    def set_qiskit_backend_options(self, qiskit_backend_options):
        self.qiskit_backend_options = qiskit_backend_options
        return self

    def set_result_file(self, result_file):
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        self.result_file = result_file
        return self

    def set_cluster_info(self, cluster_info):
        self.cluster_info = cluster_info
        return self
    
    def set_pilot(self, pilot):
        self.pilot = pilot
        return self

    def set_simulator(self, simulator):
        self.simulator = simulator
        return self    

    def build(self, executor):
        return CircuitExecution(executor, self.depth_of_recursion, self.num_qubits, self.n_entries, self.circuit_depth,
                                self.size_of_observable, self.qiskit_backend_options, self.result_file, self.current_datetime, self.cluster_info, self.pilot, self.simulator)


class CircuitExecution(Motif):
    def __init__(self, executor, depth_of_recursion, num_qubits, n_entries, circuit_depth, size_of_observable,
                 qiskit_backend_options, result_file, timestamp, cluster_info, pilot, simulator):
        super().__init__(executor, num_qubits)
        self.depth_of_recursion = depth_of_recursion
        self.n_entries = n_entries
        self.circuit_depth = circuit_depth
        self.size_of_observable = size_of_observable
        self.qiskit_backend_options = qiskit_backend_options
        self.result_file = result_file
        self.timestamp = timestamp
        self.cluster_info = cluster_info
        self.pilot = pilot
        self.simulator = simulator
        header = ["timestamp", "num_qubits", "n_entries", "circuit_depth", "size_of_observable", "depth_of_recursion",
                  "compute_time_sec", "quantum_options", "cluster_info"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)

    
    def submit_tasks(self):
        circuits, observables = generate_data(
            depth_of_recursion=1,
            num_qubits=self.num_qubits,
            n_entries=self.n_entries,
            circuit_depth=self.circuit_depth,
            size_of_observable=self.size_of_observable
        )

        circuits_observables = zip(circuits, observables)
        
        # Submit all the tasks
        futures = self.executor.submit_tasks(run_circuit, circuits_observables, self.qiskit_backend_options, self.simulator, pilot=self.pilot)
        
        return futures
    
    def wait(self, futures):
        # wait for the tasks to complete        
        start_time = time.time()
        self.executor.wait(futures)
        end_time = time.time()
        compute_time_ms = end_time-start_time
        self.metrics_file_writer.write([self.timestamp, self.num_qubits, self.n_entries, self.circuit_depth,
                                        self.size_of_observable, self.depth_of_recursion,
                                        compute_time_ms, str(self.qiskit_backend_options), str(self.cluster_info)])

        self.metrics_file_writer.close()




SIZE_OF_OBSERVABLE = "size_of_observable"
CIRCUIT_DEPTH = "circuit_depth"
NUM_ENTRIES = "num_entries"
QUBITS = "qubits"
QISKIT_BACKEND_OPTIONS = "qiskit_backend_options"
