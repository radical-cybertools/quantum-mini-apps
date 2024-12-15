import os
from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.motifs.dist_state_vector_motif import DistStateVector
import json


class QuantumSimulation:
    def __init__(self, cluster_config, parameters=None):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters
        self.parameters_path = os.path.join(os.getcwd(), 'parameters.json')
        # Create a JSON file with the parameter values
        with open(self.parameters_path, 'w') as file:
            json.dump(self.parameters, file)

    def run(self):        
        sv = DistStateVector(self.executor, self.parameters_path)
        sv.run()


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
            "cores_per_node": 64,
            "gpus_per_node": 4,
            "queue": "premium",
            "walltime": 30,            
            "project": "m4408",
            "scheduler_script_commands": ["#SBATCH --constraint=cpu"],
            # "scheduler_script_commands": ["#SBATCH --constraint=gpu",
            #                                 "#SBATCH --gpus-per-task=1",
            #                                 "#SBATCH --ntasks-per-node=4",
            #                                 "#SBATCH --gpu-bind=none"],
        }
    }


    sv_parameters = {
        "num_runs": 2,
        "n_wires": 10,
        "n_layers": 2,
        "diff_method": "adjoint",
        "pennylane_device_config": {
            "name": 'lightning.qubit',
            "wires": 10,
        }
    }
    qs = QuantumSimulation(cluster_info, sv_parameters)    
    try:            
        qs.run()            
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        qs.executor.close()
