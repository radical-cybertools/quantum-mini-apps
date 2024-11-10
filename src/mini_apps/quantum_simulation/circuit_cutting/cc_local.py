import os

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.motifs.circuit_cutting_motif import (
    BASE_QUBITS, OBSERVABLES, SCALE_FACTOR, SUB_CIRCUIT_TASK_RESOURCES, SUBCIRCUIT_SIZE, FULL_CIRCUIT_TASK_RESOURCES,
    CircuitCuttingBuilder)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class QuantumSimulation:
    def __init__(self, cluster_config, parameters=None):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters

    def run(self):
        cc_builder = CircuitCuttingBuilder()
        ce = cc_builder.set_subcircuit_size(self.parameters[SUBCIRCUIT_SIZE]) \
            .set_base_qubits(self.parameters[BASE_QUBITS]) \
            .set_observables(self.parameters[OBSERVABLES]) \
            .set_scale_factor(self.parameters[SCALE_FACTOR]) \
            .set_result_file(os.path.join(SCRIPT_DIR, "result.csv")) \
            .set_sub_circuit_task_resources(self.parameters[SUB_CIRCUIT_TASK_RESOURCES]) \
            .set_full_circuit_task_resources(self.parameters[FULL_CIRCUIT_TASK_RESOURCES]) \
            .build(self.executor)

        ce.run()


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

    cc_parameters = {
        SUBCIRCUIT_SIZE : 2,
        BASE_QUBITS: 7,
        SCALE_FACTOR : 1,
        OBSERVABLES: ["ZIIIIII", "IIIZIII", "IIIIIII"], 
        SUB_CIRCUIT_TASK_RESOURCES : {'num_cpus': 1, 'num_gpus': 0, 'memory': None},
        FULL_CIRCUIT_TASK_RESOURCES : {'num_cpus': 1, 'num_gpus': 0, 'memory': None}
    }

    qs = QuantumSimulation(cluster_info, cc_parameters)
    qs.run()