import dask.bag as db
from qiskit_aer.primitives import Estimator as AirEstimator

from motifs.base_motif import Motif
from motifs.qiskit_benchmark import generate_data


def run_circuit(circ_obs, qiskit_backend_options):
    estimator_result = AirEstimator(backend_options=qiskit_backend_options).run(circ_obs[0], circ_obs[1]).result()
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

    def build(self, executor):
        return CircuitExecution(executor, self.depth_of_recursion, self.num_qubits, self.n_entries, self.circuit_depth,
                                self.size_of_observable, self.qiskit_backend_options)


class CircuitExecution(Motif):
    def __init__(self, executor, depth_of_recursion, num_qubits, n_entries, circuit_depth, size_of_observable,
                 qiskit_backend_options):
        super().__init__(executor, num_qubits)
        self.depth_of_recursion = depth_of_recursion
        self.n_entries = n_entries
        self.circuit_depth = circuit_depth
        self.size_of_observable = size_of_observable
        self.qiskit_backend_options = qiskit_backend_options


    def run(self):
        circuits, observables = generate_data(
            depth_of_recursion=1,
            num_qubits=self.num_qubits,
            n_entries=self.n_entries,
            circuit_depth=self.circuit_depth,
            size_of_observable=self.size_of_observable
        )

        circuits_observables = zip(circuits, observables)
        circuit_bag = db.from_sequence(circuits_observables)

        # Submit all the tasks
        futures = self.executor.submit_tasks(circuit_bag, run_circuit, self.qiskit_backend_options)

        # wait for the tasks to complete
        self.executor.wait(futures)


SIZE_OF_OBSERVABLE = "size_of_observable"
CIRCUIT_DEPTH = "circuit_depth"
NUM_ENTRIES = "num_entries"
QUBITS = "qubits"
