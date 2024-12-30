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
        self.sv = None

    def update_parameters(self, parameters):
        self.parameters = parameters
        if self.sv is None:
            self.sv = DistStateVector(self.executor, self.parameters)
            

    def run(self):        
        self.sv = DistStateVector(self.executor, self.parameters)
        self.sv.run()


# Define benchmark configurations at the top
BENCHMARK_CONFIG = {
    'num_runs': 3,
    'hardware_configs': [
        {
            'nodes': [16],
            'cores_per_node': 128,
            'gpus_per_node': [4]
        }
    ],
    'circuit_configs': [
        {
            'qubit_sizes': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
            'enable_jacobian': [False, True],
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
            "walltime": 180,            
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
        
    qs = None 
    try:
        # Iterate over hardware configurations
        for hw_config in BENCHMARK_CONFIG['hardware_configs']:
            for nodes in hw_config['nodes']:
                for gpus in hw_config['gpus_per_node']:
                    # Update cluster configuration
                    cluster_info = create_cluster_info_perlmutter(nodes, cores=128, gpus=gpus)
                    try:
                        qs = QuantumSimulation(cluster_info, parameters=None)
                        # Iterate over circuit configurations
                        for circuit_config in BENCHMARK_CONFIG['circuit_configs']:
                            for qubit_size in circuit_config['qubit_sizes']:
                                for enable_jacobian in circuit_config['enable_jacobian']:
                                    # Update quantum simulation parameters
                                    sv_parameters = {
                                        "num_runs": BENCHMARK_CONFIG['num_runs'],
                                        "n_wires": qubit_size,
                                        "n_layers": 2,
                                        "enable_jacobian": enable_jacobian,
                                        "diff_method": "adjoint",
                                        "enable_qjit": False,
                                        "pennylane_device_config": {
                                            "device": 'lightning.gpu',
                                            "mpi": "True"
                                        }
                                    }
                                    
                                    qs.update_parameters(sv_parameters)

                                    print(f"Running with {nodes} nodes, {gpus} GPUs, {qubit_size} qubits, enable_jacobian={enable_jacobian}")
                                    
                                    try:            
                                        qs.run()            
                                    except Exception as e:
                                        print(f"Error @ {qubit_size} qubits (enable_jacobian: ): {e}")                                    
                                    finally:
                                        pass                   
                    except Exception as e:
                        print(f"Error: {e}")        
                    finally:     
                        qs.executor.close()
    except Exception as e:
        print(f"Error: {e}")        
    finally:
        pass
