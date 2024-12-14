import os
import sys
import math
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.motifs.circuit_cutting_motif import (
    BASE_QUBITS, NUM_SAMPLES, OBSERVABLES, SCALE_FACTOR, SIMULATOR_BACKEND_OPTIONS, SUB_CIRCUIT_TASK_RESOURCES, SUBCIRCUIT_SIZE, FULL_CIRCUIT_TASK_RESOURCES,
    CircuitCuttingBuilder)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# import pdb


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
            .set_result_file(os.path.join(SCRIPT_DIR, f"result_{timestamp}.csv")) \
            .set_sub_circuit_task_resources(self.parameters[SUB_CIRCUIT_TASK_RESOURCES]) \
            .set_full_circuit_task_resources(self.parameters[FULL_CIRCUIT_TASK_RESOURCES]) \
            .set_num_samples(self.parameters[NUM_SAMPLES]) \
            .build(self.executor)

        # pdb.set_trace()
        cc.run()


if __name__ == "__main__":
    RESOURCE_URL_HPC = "ssh://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")
    
    circuit_sizes = [8, 16, ]
    subcircuit_sizes = {circuit_size: [size for size in range(2, circuit_size // 2 + 1, 2)] for circuit_size in circuit_sizes}

    for circuit_size in circuit_sizes:
        for subcircuit_size in subcircuit_sizes[circuit_size]:
            try:
                cluster_info = {       
                    "executor": "pilot",
                    "config": {
                        "resource": RESOURCE_URL_HPC,
                        "working_directory": WORKING_DIRECTORY,
                        "type": "ray",
                        "number_of_nodes": 1,
                        "cores_per_node": 10,
                        "gpus_per_node": 4,
                    }
                }

                cc_parameters = {
                    SUBCIRCUIT_SIZE : subcircuit_size,
                    BASE_QUBITS: circuit_size,
                    SCALE_FACTOR : 1,
                    OBSERVABLES:  ["Z" + "I" * (circuit_size - 1)], # ["ZIIIIII", "IIIZIII", "IIIIIII"], 
                    NUM_SAMPLES: 100,
                    SUB_CIRCUIT_TASK_RESOURCES : {'num_cpus': 1, 'num_gpus': 0, 'memory': None},
                    FULL_CIRCUIT_TASK_RESOURCES : {'num_cpus': 1, 'num_gpus': 0, 'memory': None},
                    # SIMULATOR_BACKEND_OPTIONS: {"backend_options": {"shots": 4096, "device":"GPU", "method":"statevector", "blocking_enable":True, "batched_shots_gpu":True, "blocking_qubits":25}}
                }
                qs = QuantumSimulation(cluster_info, cc_parameters)
                qs.run()
                #qs.close()
            except Exception as e:
                print(f"Error: {e}")
                raise e
