import os
from executor.manager import MiniAppExecutor
from motifs.circuit_execution_motif import CircuitExecutionBuilder, SIZE_OF_OBSERVABLE, CIRCUIT_DEPTH, \
    NUM_ENTRIES, QUBITS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class QuantumSimulation:
    def __init__(self, cluster_config, parameters=None):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters

    def run(self):
        ce_builder = CircuitExecutionBuilder()
        ce = ce_builder.set_num_qubits(self.parameters[QUBITS]) \
            .set_n_entries(self.parameters[NUM_ENTRIES]) \
            .set_circuit_depth(self.parameters[CIRCUIT_DEPTH]) \
            .set_size_of_observable(self.parameters[SIZE_OF_OBSERVABLE]) \
            .set_result_file(os.path.join(SCRIPT_DIR, "result.csv")) \
            .build(self.executor)

        ce.run()


if __name__ == "__main__":
    cluster_info = {
        "executor": "dask",
        "config": {
            "type": "quantum-simulation",
            "quantum-simulation": {
                "n_workers": 4,
                "threads_per_worker": 2,
                "memory_limit": "4GB"
            }
        }

    }

    ce_parameters = {
        QUBITS: 10,
        NUM_ENTRIES: 10,
        CIRCUIT_DEPTH: 1,
        SIZE_OF_OBSERVABLE: 1
    }

    qs = QuantumSimulation(cluster_info, ce_parameters)
    qs.run()
