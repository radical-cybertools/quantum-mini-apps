import os
import sys
import logging
from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.distributed_state_vector.motif import DistStateVector
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

class QuantumSimulation:
    def __init__(self, cluster_config, parameters=None):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters
        

    def run(self):        
        sv = DistStateVector(self.executor, self.parameters)
        sv.run()


# Define benchmark configurations at the top
BENCHMARK_CONFIG = {
    'num_runs': 1,
    'hardware_configs': [
        {
            'nodes': [2],
            'cores_per_node': 128,
            'gpus_per_node': [4]
        }
    ],
    'circuit_configs': [
        {
            'qubit_sizes': [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
            'enable_jacobian': [False],
        }
    ]
}


def create_cluster_info_perlmutter(nodes, cores=128, gpus=4):
    return {       
        "executor": "pilot",
        "config": {
            "resource": RESOURCE_URL_HPC,
            "working_directory": WORKING_DIRECTORY,
            "type": "ray",
            "number_of_nodes": nodes,
            "cores_per_node": cores,
            "gpus_per_node": gpus,
            "queue": "premium",
            #"queue": "regular",
            "walltime": 59,            
            "project": "m4408",
            "scheduler_script_commands": ["#SBATCH --constraint=gpu&hbm80g",
                                            "#SBATCH --gpus-per-task=1",
                                            f"#SBATCH --ntasks-per-node={gpus}",
                                            "#SBATCH --gpu-bind=none"],
            }
        }


if __name__ == "__main__":
    RESOURCE_URL_HPC = "slurm://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

    # Iterate over hardware configurations
    for hw_config in BENCHMARK_CONFIG['hardware_configs']:
        for nodes in hw_config['nodes']:
            for gpus in hw_config['gpus_per_node']:
                # Update cluster configuration
                cluster_info = create_cluster_info_perlmutter(nodes, cores=128, gpus=gpus)

                # Iterate over circuit configurations
                for circuit_config in BENCHMARK_CONFIG['circuit_configs']:
                    for qubit_size in circuit_config['qubit_sizes']:
                        for enable_jacobian in circuit_config['enable_jacobian']:
                            # Update quantum simulation parameters
                            sv_parameters = {
                                "num_runs": BENCHMARK_CONFIG['num_runs'],
                                "n_wires": qubit_size,
                                "n_layers": 1,
                                "enable_jacobian": enable_jacobian,
                                "diff_method": "adjoint",
                                "pennylane_device_config": {
                                    "device": 'lightning.gpu',
                                    "mpi": "True"
                                }
                            }

                            print(f"Running with {nodes} nodes, {gpus} GPUs, {qubit_size} qubits, enable_jacobian={enable_jacobian}")
                            qs = QuantumSimulation(cluster_info, sv_parameters)    
                            try:            
                                qs.run()            
                            except Exception as e:
                                print(f"Error: {e}")
                                raise e
                            finally:
                                qs.executor.close()
