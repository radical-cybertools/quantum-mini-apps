import os

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.motifs.circuit_execution_motif import CircuitExecutionBuilder, SIZE_OF_OBSERVABLE, CIRCUIT_DEPTH, \
    NUM_ENTRIES, QUBITS, QISKIT_BACKEND_OPTIONS


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class QuantumSimulation:
    def __init__(self, cluster_config):
        self.executor = MiniAppExecutor(cluster_config).get_executor()

    def submit_circuits(self, parameters, pilot=None):
        ce_builder = CircuitExecutionBuilder()
        ce = ce_builder.set_num_qubits(parameters[QUBITS]) \
            .set_n_entries(parameters[NUM_ENTRIES]) \
            .set_circuit_depth(parameters[CIRCUIT_DEPTH]) \
            .set_size_of_observable(parameters[SIZE_OF_OBSERVABLE]) \
            .set_qiskit_backend_options(parameters[QISKIT_BACKEND_OPTIONS]) \
            .set_result_file(os.path.join(SCRIPT_DIR, "result.csv")) \
            .build(self.executor)
                    
        return self.ce.submit_tasks()
        
    def wait(self, futures):
        self.ce.wait(futures)
    
    
    def close(self):
        self.executor.close()  
        
if __name__ == "__main__":
    scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler_file.json")
    cluster_info = {
        "executor": "dask",
        "config": {
            "scheduler_file": scheduler_file
        }
    }

    ce_parameters = {
        QUBITS: 25,
        NUM_ENTRIES: 1024,
        CIRCUIT_DEPTH: 1,
        SIZE_OF_OBSERVABLE: 1,
        QISKIT_BACKEND_OPTIONS: {"method": "statevector", "device": 'GPU', "cuStateVec_enable": True, "shots": None}
    }

    qs = QuantumSimulation(cluster_info, ce_parameters)
    futures = qs.submit_circuits(ce_parameters)
    qs.wait(futures)
