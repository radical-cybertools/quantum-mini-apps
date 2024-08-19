import os

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.motifs.circuit_execution_motif import CircuitExecutionBuilder, SIZE_OF_OBSERVABLE, CIRCUIT_DEPTH, \
    NUM_ENTRIES, QUBITS, QISKIT_BACKEND_OPTIONS


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
            .set_qiskit_backend_options(self.parameters[QISKIT_BACKEND_OPTIONS]) \
            .set_result_file(os.path.join(SCRIPT_DIR, "result.csv")) \
            .build(self.executor)

        ce.run()


if __name__ == "__main__":

    RESOURCE_URL_HPC = "slurm://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

    cluster_info = {       
        "executor": "pilot",
        "config": {
            "resource": RESOURCE_URL_HPC,
            "working_directory": WORKING_DIRECTORY,
            "type": "dask",
            "number_of_nodes": 1,
            "cores_per_node": 2,
            "gpus_per_node": 0,
            "queue": "debug",
            "walltime": 30,            
            "project": "m4408",
            "conda_environment": "/pscratch/sd/l/luckow/conda/quantum-mini-apps2",
            "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
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
    qs.run()
