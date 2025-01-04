import os

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.circuit_execution.motifs.circuit_execution_motif import CircuitExecutionBuilder, SIZE_OF_OBSERVABLE, CIRCUIT_DEPTH, \
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
            .set_qiskit_backend_options(parameters[QISKIT_BACKEND_OPTIONS]) \
            .set_cluster_info(cluster_info) \
            .set_result_file(os.path.join(SCRIPT_DIR, "results", "results_multi_pilot.csv")) \
            .set_pilot(pilot) \
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
    
    # Loop to iterate over different numbers of qubits
    # Create a list of qubit values: powers of two up to 16, then increments of 4 from 20 to 28
    qubit_values = [2**i for i in range(1, 5)] + list(range(20, 30, 2))
   
    for nodes in [1]:
        for cores_per_node in [10]:
            try:
                cluster_info = {  
                    "executor": "multi-pilot",
                    "type": "dask",
                    "working_directory": WORKING_DIRECTORY,
                    "config": { 
                        "cpu-pilot": {
                            "resource": RESOURCE_URL_HPC,                            
                            "number_of_nodes": 1,
                            "cores_per_node": cores_per_node,
                            "gpus_per_node": 0,
                            "queue": "debug",
                            "walltime": 30,            
                            "project": "m4408",
                            "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
                        },
                        "gpu-pilot": {
                            "resource": RESOURCE_URL_HPC,
                            "number_of_nodes": 1,
                            "cores_per_node": cores_per_node,
                            "gpus_per_node": 0,
                            "queue": "debug",
                            "walltime": 30,            
                            "project": "m4408",
                            "scheduler_script_commands": ["#SBATCH --constraint=cpu"]                            
                        },
                        "ionq-pilot": {
                            "resource": RESOURCE_URL_HPC,
                            "number_of_nodes": 1,
                            "cores_per_node": cores_per_node,
                            "gpus_per_node": 0,
                            "queue": "debug",
                            "walltime": 30,            
                            "project": "m4408",
                            "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
                        }                            
                    }, 
                }

                qs = QuantumSimulation(cluster_info)
            
                for qubits in [10]:
                    ce_parameters = {
                        QUBITS: qubits,  # Adjust the number of qubits dynamically
                        NUM_ENTRIES: 10,
                        CIRCUIT_DEPTH: 1,
                        SIZE_OF_OBSERVABLE: 1,
                        "SIMULATOR": "aer_simulator",
                        QISKIT_BACKEND_OPTIONS: {
                            "method": "statevector",
                            "device": 'CPU',
                            "cuStateVec_enable": False,
                            "shots": None,
                        }
                    }
                    
                    aer_backend_futures = qs.submit_circuits(ce_parameters, pilot="cpu-pilot")
                    
                    # update backend to ionq
                    ce_parameters["SIMULATOR"] = "ionq_simulator"
                    ionq_backend_futures = qs.submit_circuits(ce_parameters, pilot="ionq-pilot")
                                        
                    # update backend to cpu-pilot
                    ce_parameters["SIMULATOR"] = "aer_simulator"
                    ce_parameters[QISKIT_BACKEND_OPTIONS]["device"] = "CPU"                    
                    gpu_backend_futures = qs.submit_circuits(ce_parameters, pilot="gpu-pilot")
                    
                    qs.wait(aer_backend_futures + ionq_backend_futures + gpu_backend_futures)                    
                qs.close()
            except Exception as e:
                print(e)
                continue