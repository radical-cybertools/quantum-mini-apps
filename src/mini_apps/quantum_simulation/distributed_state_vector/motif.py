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
from datetime import datetime

# Third party imports
import pennylane as qml

# import jax.numpy as jnp
from pennylane import qjit 
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
        
        # set startup for external tasks to later measure the overhead
        self.parameters["agent_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        enable_qjit = parameters.get("enable_qjit", False)
        if diff_method == "None":
            diff_method = None
        if pennylane_device_config["mpi"].lower() == "true":
            pennylane_device_config["mpi"] = True
        else:
            pennylane_device_config["mpi"] = False
        
        # for computation of task startup overhead
        start_time_agent_str = parameters.get("agent_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        start_time_agent = datetime.strptime(start_time_agent_str, "%Y%m%d_%H%M%S").timestamp()        
        start_time_process = time.time()
        mpi_startup_time = start_time_process - start_time_agent

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
            #return qml.expval(qml.PauliZ(0))

        params = np.array(
            np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)), 
            dtype=np.float64
        )

        if enable_jacobian:
            print(f"Initializing QNode with jacobian enabled: interface=autograd, diff_method={diff_method}")
            circuit_adj = qml.qnode(dev, interface="autograd", diff_method=diff_method)(circuit_adj)
        else:
            print("Initializing QNode without jacobian")
            circuit_adj = qml.qnode(dev, diff_method=None)(circuit_adj)
        
        if enable_qjit:
            circuit_adj = qjit(circuit_adj)

        # Set trainable parameters for calculating circuit Jacobian at the rank=0 process
        if rank == 0:
            params = np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
        else:
            params = None

        # Broadcast the trainable parameters across MPI processes from rank=0 process
        params = comm.bcast(params, root=0)

        # Create MetricsWriter instance if rank 0
        if rank == 0:
            # Create results directory if it doesn't exist
            results_dir = os.path.abspath("results")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            print(f"Results directory: {results_dir}")

            # Create a timestamp and initialize the MetricsFileWriter with the path to the results directory            
            metrics_writer = MetricsFileWriter(
                os.path.join(results_dir, f"distributed_state_vector_{start_time_agent_str}.csv"),
                header=[
                    "timestamp",
                    "num_gpus",
                    "wires",
                    "layers", 
                    "time",
                    "expval",
                    "enable_jacobian",
                    "enable_qjit",
                    "mpi_startup_time"
                ]
            )
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
            runtime = end - start
            timing.append(runtime)
            
            if rank == 0:                
                metrics = [
                    start_time_agent_str,
                    size,
                    n_wires,
                    n_layers,
                    runtime,
                    str(result[0])[:10],
                    enable_jacobian,
                    enable_qjit,
                    mpi_startup_time
                ]
                metrics_writer.write(metrics)
                print("timestamp: ", start_time_agent_str, " num_gpus: ", size, " wires: ", n_wires, 
                      " layers ", n_layers, " time: ", runtime, " result q0: ", result[0],
                      " enable_qjit: ", enable_qjit, " mpi: ", pennylane_device_config["mpi"],
                      " mpi_startup_time: ", mpi_startup_time
                      )
        
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
    parser.add_argument('--device', type=str, default='lightning.gpu',
                      help='PennyLane device name (default: lightning.gpu)')
    parser.add_argument('--mpi', type=str, default='True',
                      choices=['True', 'False'],
                      help='Enable MPI (default: True)')
    parser.add_argument('--enable-jacobian', type=str, default='False',
                      choices=['True', 'False'],
                      help='Enable Jacobian calculation (default: False)')
    parser.add_argument('--batch-obs', type=str, default='False',
                      choices=['True', 'False'],
                      help='Enable batch observations (default: False)')
    parser.add_argument('--enable-qjit', type=str, default='False',
                      choices=['True', 'False'],
                      help='Enable QJIT compilation (default: True)')
    parser.add_argument('--agent-timestamp', type=str, 
                  default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                  help='Timestamp for logging (default: current time in format YYYYMMDD_HHMMSS)')
    
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
            "enable_jacobian": args.enable_jacobian.lower() == 'true',
            "enable_qjit": args.enable_qjit.lower() == 'true',
            "agent_timestamp": args.agent_timestamp,
            "pennylane_device_config": {
                "name": args.device,
                "mpi": args.mpi,
                "batch_obs": args.batch_obs.lower() == 'true'
            }
        }
        dist_state_vector = DistStateVector(None, parameters)
    
    # Run the simulation
    dist_state_vector.run()
        
