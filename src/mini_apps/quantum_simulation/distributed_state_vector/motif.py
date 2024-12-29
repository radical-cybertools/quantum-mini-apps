# Part of is taken from Pennylane: https://pennylane.ai/blog/2023/09/distributing-quantum-simulations-using-lightning-gpu-with-NVIDIA-cuQuantum

# Standard library imports
import argparse
import json
import os
import subprocess
import sys
import time
from timeit import default_timer as timer
import logging

# Third party imports
import pennylane as qml
from mpi4py import MPI
from pennylane import numpy as np

from engine.base.base_motif import Motif
from engine.metrics.csv_writer import MetricsFileWriter


RUN_SCRIPT = os.path.join(os.path.dirname(__file__))

RAY_TASK_RESOURCES = {
            "num_cpus": 1,
            "num_gpus": 1,            
            "memory": None,
        }

class DistStateVector(Motif):    

    def __init__(self, executor, parameters):
        """
        Initialize with either a parameters dictionary or a path to a JSON file
        """
        super().__init__(executor)
        if isinstance(parameters, str):
            self.parameters = self._load_parameters_from_file(parameters)
        else:
            self.parameters = parameters

        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Check if the logger already has handlers to prevent duplicates
        if not logger.hasHandlers():
            # Create a console handler and set the log level
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create a formatter and add it to the console handler
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)

            # Add the console handler to the logger
            logger.addHandler(console_handler)

        self.logger = logger
    
    def _load_parameters_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}")
            sys.exit(1)
        
    def run(self):
        """ Called from mini_app.py to generate srun or mpirun command"""

        # Check if MPI is enabled in the device config
        mpi_enabled = self.parameters.get("pennylane_device_config", {}).get("mpi", "").lower() == "true"
        
        num_nodes = 1
        num_gpus = 0
        if mpi_enabled and self.executor is not None:
            self.logger.info(f"Running simulation via Pilot/MPI with parameters: {self.parameters}")
            # Serialize the circuit and observable to files
            # Get working directory from executor or use current directory as fallback
            #working_dir = self.executor.cluster_config["config"]["working_directory"]
            
            num_nodes = self.executor.cluster_config.get("config", {}).get("number_of_nodes", 1)
            num_gpus_per_node = self.executor.cluster_config.get("config", {}).get("gpus_per_node", 0)
            num_gpus = num_nodes * num_gpus_per_node
    
            # Submit task for MPI parallel execution via command line
            cmd = ["srun", "-N", str(num_nodes), 
                   f"-n {num_gpus}", "python",  
                   sys.modules[self.__class__.__module__].__file__]

                        # Add parameters as command line arguments
            if isinstance(self.parameters, dict):
                for key, value in self.parameters.items():
                    if key == "pennylane_device_config":
                        for device_key, device_value in value.items():
                            cmd.extend([f"--{device_key}", str(device_value)])
                    else:
                        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            task = self.executor.submit_task(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                resources=RAY_TASK_RESOURCES
            )
            result = self.executor.get_results([task])[0]
        
            if result.returncode != 0:
                raise RuntimeError(f"Command failed with error: {result.stderr}")
                       
            # Log the number of output lines
            output_lines = result.stdout.strip().split('\n')
            self.logger.info(f"Number of output lines: {len(output_lines)}")
            # Log the complete stdout
            self.logger.info(f"Complete output:\n{result.stdout}")
        
        else:
            # Run directly in process if MPI is not enabled
            try:
                self.logger.info(f"Running simulation in_process with parameters: {self.parameters}")
                self.run_simulation(self.parameters)
            except Exception as e:
                self.logger.error(f"Simulation failed with error: {str(e)}")
                self.logger.exception("Full traceback:")
                raise RuntimeError(f"Simulation failed: {str(e)}") from e

    @staticmethod
    def run_simulation(parameters):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        pennylane_device_config = parameters["pennylane_device_config"]
        num_runs = parameters["num_runs"]
        n_layers = parameters["n_layers"]
        n_wires = parameters["n_wires"]
        pennylane_device_config["wires"] = n_wires
        diff_method = parameters["diff_method"]
        enable_jacobian = parameters.get("enable_jacobian", False)
        if diff_method == "None":
            diff_method = None
        if pennylane_device_config["mpi"].lower() == "true":
            pennylane_device_config["mpi"] = True
        else:
            pennylane_device_config["mpi"] = False

        # Instantiate CPU (lightning.qubit) or GPU (lightning.gpu) device
        # mpi=True to switch on distributed simulation
        # batch_obs=True to reduce the device memory demand for adjoint backpropagation

        # Print device configuration with rank information
        if rank == 0:
            print(f"Initializing device with configuration:\n{json.dumps(pennylane_device_config, indent=2)}")
        
        dev = qml.device(**pennylane_device_config)

        # Create QNode of device and circuit
        def circuit_adj(weights):
            qml.StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
            return qml.math.hstack([qml.expval(qml.PauliZ(i)) for i in range(n_wires)])

        if enable_jacobian:
            print(f"Initializing QNode with jacobian enabled: interface=autograd, diff_method={diff_method}")
            circuit_adj = qml.qnode(dev, interface="autograd", diff_method=diff_method)(circuit_adj)
        else:
            print("Initializing QNode without jacobian")
            circuit_adj = qml.qnode(dev)(circuit_adj)

        # Set trainable parameters for calculating circuit Jacobian at the rank=0 process
        if rank == 0:
            params = np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
        else:
            params = None

        # Broadcast the trainable parameters across MPI processes from rank=0 process
        params = comm.bcast(params, root=0)

        # Create MetricsWriter instance if rank 0
        if rank == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            metrics_writer = MetricsFileWriter(f"distributed_state_vector_{timestamp}.csv", header=[
                "timestamp",
                "num_gpus",
                "wires",
                "layers", 
                "time",
                "enable_jacobian"
            ])
        timing = []
        for t in range(num_runs):
            start = time.time()            
            if enable_jacobian:
                print("Calculating Jacobian")
                result = qml.jacobian(circuit_adj)(params)
            else:
                print("Calculating State Vector without Jacobian")
                result = circuit_adj(params)
            end = time.time()
            timing.append(end - start)
            if rank == 0:
                current_timestamp = time.strftime("%Y%m%d-%H%M%S")
                metrics = [
                    current_timestamp,
                    size,
                    n_wires,
                    n_layers,
                    qml.numpy.mean(timing),
                    enable_jacobian
                ]
                metrics_writer.write(metrics)
                print("timestamp: ", current_timestamp, " num_gpus: ", size, " wires: ", n_wires, " layers ", n_layers, " time: ", qml.numpy.mean(timing))
        
        if rank == 0:
            metrics_writer.close()

        # MPI barrier to ensure all calculations are done
        comm.Barrier()

    
        
if __name__ == "__main__":
    """ Run script from command line via mpirun or srun on Perlmutter or SLURM managed clusters
    """


    parser = argparse.ArgumentParser(description='Distributed State Vector Simulation')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', type=str,
                      help='Path to JSON configuration file')
    group.add_argument('--use-cli', action='store_true',
                      help='Use command line parameters instead of config file')
    
    # Command line parameters
    parser.add_argument('--num-runs', type=int, default=2,
                      help='Number of runs (default: 2)')
    parser.add_argument('--n-wires', type=int, default=10,
                      help='Number of wires (default: 10)')
    parser.add_argument('--n-layers', type=int, default=2,
                      help='Number of layers (default: 2)')
    parser.add_argument('--diff-method', type=str, default='None',
                      choices=['adjoint', 'parameter-shift', 'None'],
                      help='Differentiation method (default: adjoint)')
    parser.add_argument('--device', type=str, default='lightning.qpu',
                      help='PennyLane device name (default: lightning.qpu)')
    parser.add_argument('--mpi', type=str, default='True',
                      choices=['True', 'False'],
                      help='Enable MPI (default: True)')
    parser.add_argument('--enable-jacobian', type=str, default='False',
                      choices=['True', 'False'],
                      help='Enable Jacobian calculation (default: False)')
    parser.add_argument('--batch-obs', type=str, default='False',
                      choices=['True', 'False'],
                      help='Enable batch observations (default: False)')
    
    args = parser.parse_args()

    if args.config:
        # Use JSON config file
        dist_state_vector = DistStateVector(None, args.config)
        
    else:
        # Use command line parameters
        parameters = {
            "num_runs": args.num_runs,
            "n_wires": args.n_wires,
            "n_layers": args.n_layers,
            "diff_method": args.diff_method,
            "enable_jacobian": args.enable_jacobian.lower() == 'true',  # Convert string to boolean
            "pennylane_device_config": {
                "name": args.device,
                "mpi": args.mpi,
                "batch_obs": args.batch_obs.lower() == 'true'  # Convert string to boolean
            }
        }
        dist_state_vector = DistStateVector(None, parameters)
    
    # Run the simulation
    dist_state_vector.run()
        
