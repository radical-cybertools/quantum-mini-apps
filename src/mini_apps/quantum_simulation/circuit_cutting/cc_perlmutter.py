import os

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.motifs.circuit_cutting_motif import (
    BASE_QUBITS, OBSERVABLES, SCALE_FACTOR, SIMULATOR_BACKEND_OPTIONS, SUB_CIRCUIT_TASK_RESOURCES, SUBCIRCUIT_SIZE, FULL_CIRCUIT_TASK_RESOURCES,
    CircuitCuttingBuilder)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class QuantumSimulation:
    def __init__(self, cluster_config, parameters=None):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters

    def run(self):
        cc_builder = CircuitCuttingBuilder()
        cc = cc_builder.set_subcircuit_size(self.parameters[SUBCIRCUIT_SIZE]) \
            .set_base_qubits(self.parameters[BASE_QUBITS]) \
            .set_observables(self.parameters[OBSERVABLES]) \
            .set_scale_factor(self.parameters[SCALE_FACTOR]) \
            .set_result_file(os.path.join(SCRIPT_DIR, "result.csv")) \
            .set_sub_circuit_task_resources(self.parameters[SUB_CIRCUIT_TASK_RESOURCES]) \
            .set_full_circuit_task_resources(self.parameters[FULL_CIRCUIT_TASK_RESOURCES]) \
            .set_qiskit_backend_options(self.parameters[SIMULATOR_BACKEND_OPTIONS]) \
            .build(self.executor)

        cc.run()


if __name__ == "__main__":
    RESOURCE_URL_HPC = "slurm://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")
    nodes = [1]    
    for node in nodes:
        try:
            cluster_info = {       
                "executor": "pilot",
                "config": {
                    "resource": RESOURCE_URL_HPC,
                    "working_directory": WORKING_DIRECTORY,
                    "type": "ray",
                    "number_of_nodes": node,
                    "cores_per_node": 64,
                    "gpus_per_node": 4,
                    "queue": "premium",
                    "walltime": 30,            
                    "project": "m4408",
                    "scheduler_script_commands": ["#SBATCH --constraint=gpu",
                                                  "#SBATCH --gpus-per-task=1",
                                                  "#SBATCH --ntasks-per-node=4",
                                                  "#SBATCH --gpu-bind=none"],
                }
            }

            cc_parameters = {
                SUBCIRCUIT_SIZE : 2,
                BASE_QUBITS: 7,
                SCALE_FACTOR : 1,
                OBSERVABLES: ["ZIIIIII", "IIIZIII", "IIIIIII"], 
                SUB_CIRCUIT_TASK_RESOURCES : {'num_cpus': 1, 'num_gpus': 1, 'memory': None},
                FULL_CIRCUIT_TASK_RESOURCES : {'num_cpus': 64, 'num_gpus': 1, 'memory': None},
                SIMULATOR_BACKEND_OPTIONS: {"backend_options": {"shots": 4096, "device":"GPU", "method":"statevector", "blocking_enable":True, "batched_shots_gpu":True, "blocking_qubits":25}}
            }

            qs = QuantumSimulation(cluster_info, cc_parameters)
            qs.run()
        except Exception as e:
            print(f"Error: {e}")
            raise e
