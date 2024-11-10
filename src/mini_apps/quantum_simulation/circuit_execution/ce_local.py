import os

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.motifs.circuit_execution_motif import CircuitExecutionBuilder, SIZE_OF_OBSERVABLE, CIRCUIT_DEPTH, \
    NUM_ENTRIES, QUBITS, QISKIT_BACKEND_OPTIONS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class QuantumSimulation:
    def __init__(self, cluster_info):
        self.executor = MiniAppExecutor(cluster_info).get_executor()
        self.cluster_info = cluster_info        
        self.ce_builder = CircuitExecutionBuilder()

    
    def submit_circuits(self, parameters, pilot=None):
        self.ce = self.ce_builder.set_num_qubits(parameters[QUBITS]) \
            .set_n_entries(parameters[NUM_ENTRIES]) \
            .set_circuit_depth(parameters[CIRCUIT_DEPTH]) \
            .set_size_of_observable(parameters[SIZE_OF_OBSERVABLE]) \
            .set_cluster_info(cluster_info) \
            .set_result_file(os.path.join(SCRIPT_DIR, "results", "results_local_pilot.csv")) \
            .set_simulator(parameters["SIMULATOR"]) \
            .build(self.executor)
        
        return self.ce.submit_tasks()
        
    def wait(self, futures):
        self.ce.wait(futures)
    
    
    def close(self):
        self.executor.close()


if __name__ == "__main__":
    RESOURCE_URL_HPC = "ssh://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")
        
    cluster_info = {       
        "executor": "pilot",        
        "config": {
            "resource": RESOURCE_URL_HPC,
            "working_directory": WORKING_DIRECTORY,
            "type": "ray",
            "number_of_nodes": 1,
            "cores_per_node": 10
        }
    }

    ce_parameters = {
        QUBITS: 10,
        NUM_ENTRIES: 10,
        CIRCUIT_DEPTH: 1,
        SIZE_OF_OBSERVABLE: 1,
        "SIMULATOR": "aer_simulator",
        
    }


    qs = QuantumSimulation(cluster_info)
    futures = qs.submit_circuits(ce_parameters)
    qs.wait(futures)
