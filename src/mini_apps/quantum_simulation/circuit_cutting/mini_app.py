import os
import sys
import math
import datetime
import time
import logging
import psutil
import subprocess
# import pdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.circuit_cutting.motif import (
    BASE_QUBITS,
    NUM_SAMPLES,
    OBSERVABLES,
    SCALE_FACTOR,
    CIRCUIT_CUTTING_SIMULATOR_BACKEND_OPTIONS,
    FULL_CIRCUIT_SIMULATOR_BACKEND_OPTIONS,
    SUB_CIRCUIT_TASK_RESOURCES,
    SUBCIRCUIT_SIZE,
    FULL_CIRCUIT_TASK_RESOURCES,
    FULL_CIRCUIT_ONLY,
    CIRCUIT_CUTTING_ONLY,
    SCENARIO_LABEL,
    CircuitCuttingBuilder
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define benchmark configurations at the top
BENCHMARK_CONFIG = {
    'num_runs': 3,
    'hardware_configs': [
        {
            'nodes': [1],
            'cores_per_node': 64,
            'gpus_per_node': [4]
        }
    ],
    'circuit_configs': [
        {
            'qubit_sizes': [33],
            'subcircuit_sizes': [17],  # 30//4 + 1
            'num_samples': 10000
        }
    ]
}

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class QuantumSimulation:
    def __init__(self, cluster_config, parameters=None):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters

    def run(self):
        cc_builder = CircuitCuttingBuilder()
        cc = (
            cc_builder.set_subcircuit_size(self.parameters[SUBCIRCUIT_SIZE])
            .set_base_qubits(self.parameters[BASE_QUBITS])
            .set_observables(self.parameters[OBSERVABLES])
            .set_scale_factor(self.parameters[SCALE_FACTOR])
            .set_result_file(os.path.join(SCRIPT_DIR, f"result_{timestamp}.csv"))
            .set_sub_circuit_task_resources(self.parameters[SUB_CIRCUIT_TASK_RESOURCES])
            .set_full_circuit_task_resources(
                self.parameters[FULL_CIRCUIT_TASK_RESOURCES]
            )
            .set_full_circuit_only(self.parameters[FULL_CIRCUIT_ONLY])
            .set_circuit_cutting_only(self.parameters[CIRCUIT_CUTTING_ONLY])
            .set_circuit_cutting_qiskit_options(self.parameters[CIRCUIT_CUTTING_SIMULATOR_BACKEND_OPTIONS])
            .set_full_circuit_qiskit_options(self.parameters[FULL_CIRCUIT_SIMULATOR_BACKEND_OPTIONS])
            .set_num_samples(self.parameters[NUM_SAMPLES])
            .set_scenario_label(self.parameters[SCENARIO_LABEL])
            .build(self.executor)
        )

        # pdb.set_trace()
        cc.run()

    def close(self):
        self.executor.close()
        # hack terminate all agents
        # kill_processes_by_keyword("pilot.plugins.ray_v2.agent")
        # stop_ray()




def create_cluster_info(nodes, cores, gpus):
    return {
        "executor": "pilot",
        "config": {
            "resource": RESOURCE_URL_LOCAL,
            "working_directory": WORKING_DIRECTORY,
            "type": "ray",
            "number_of_nodes": nodes,
            "cores_per_node": cores,
            "gpus_per_node": gpus,
        },
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
            "walltime": 30,            
            "project": "m4408",
            "scheduler_script_commands": ["#SBATCH --constraint=gpu&hbm80g",
                                            "#SBATCH --gpus-per-task=1",
                                            f"#SBATCH --ntasks-per-node={gpus}",
                                            "#SBATCH --gpu-bind=none"],
            }
        }

def create_cc_parameters(circuit_size, subcircuit_size, num_samples, num_nodes, num_cores, num_gpus):
    # Determine if we're using MPI for full circuit simulation
    use_mpi_full_circuit = False  # Set based on your needs

    return {
        SUBCIRCUIT_SIZE: subcircuit_size,
        BASE_QUBITS: circuit_size,
        SCALE_FACTOR: 1,
        OBSERVABLES: ["Z" + "I" * (circuit_size - 1)],
        NUM_SAMPLES: num_samples,
        SUB_CIRCUIT_TASK_RESOURCES: {
            "num_cpus": 1,
            "num_gpus": 1,
            "memory": None,
        },
        FULL_CIRCUIT_TASK_RESOURCES: {
            "num_cpus": num_gpus,
            # If using MPI, set num_gpus to 0 (srun manages GPUs via CUDA_VISIBLE_DEVICES)
            # Otherwise, Ray needs to allocate GPUs for the task to access them
            "num_gpus": 0 if use_mpi_full_circuit else num_gpus,
            "mpi_ranks": num_gpus,
            "num_nodes": num_nodes,
            "memory": None,
        },
        FULL_CIRCUIT_ONLY: False,
        CIRCUIT_CUTTING_ONLY: False,
        CIRCUIT_CUTTING_SIMULATOR_BACKEND_OPTIONS: {
            #"backend_options": {"shots": 4096, "device":"CPU", "method":"statevector"},
            "backend_options": {"device":"GPU", "method":"statevector", "shots": 4096,
                              "blocking_enable":True, "batched_shots_gpu":True,
                              "blocking_qubits":23},
            "mpi": False
        },
        FULL_CIRCUIT_SIMULATOR_BACKEND_OPTIONS: {
            #"backend_options": {"shots": 4096, "device":"CPU", "method":"statevector"},
            "backend_options": {"device":"GPU", "method":"statevector", "shots": 4096,
                              #"precision": "single",  # Use FP32 to reduce memory
                              "blocking_enable": True,
                              "batched_shots_gpu": True,
                              "blocking_qubits": 29,  
                              "cuStateVec_enable": True  # Enable multi-GPU
                              },
            "mpi": use_mpi_full_circuit
        },
        SCENARIO_LABEL: f"circuit_size_{circuit_size}_subcircuit_{subcircuit_size}_samples_{num_samples}_cores_{num_cores}_nvidia_A100"
    }

def run_mini_app_benchmark():
    for run_idx in range(BENCHMARK_CONFIG['num_runs']):
        logger.info(f"Starting benchmark run {run_idx + 1}/{BENCHMARK_CONFIG['num_runs']}")
        
        for hw_config in BENCHMARK_CONFIG['hardware_configs']:
            for nodes in hw_config['nodes']:
                for gpus in hw_config['gpus_per_node']:
                    cluster_info = create_cluster_info(
                        nodes=nodes,
                        cores=hw_config['cores_per_node'],
                        gpus=gpus
                    )
                    
                    for circuit_config in BENCHMARK_CONFIG['circuit_configs']:
                        for qubit_size in circuit_config['qubit_sizes']:
                            for subcircuit_size in circuit_config['subcircuit_sizes']:
                                try:
                                    cc_parameters = create_cc_parameters(
                                        circuit_size=qubit_size,
                                        subcircuit_size=subcircuit_size,
                                        num_samples=circuit_config['num_samples'],
                                        num_nodes=nodes,
                                        num_cores=hw_config['cores_per_node'],
                                        num_gpus=gpus
                                    )
                                    
                                    logger.info(f"Running configuration: {cc_parameters[SCENARIO_LABEL]}")
                                    qs = QuantumSimulation(cluster_info, cc_parameters)
                                    qs.run()
                                    qs.close()
                                    
                                except Exception as e:
                                    logger.error(f"Error in configuration {cc_parameters[SCENARIO_LABEL]}: {e}")
                                    raise e

# Module-level constants for resource configuration
# These can be overridden by setting environment variables or modifying before import
RESOURCE_URL_HPC = os.environ.get("RESOURCE_URL_HPC", "slurm://localhost")
RESOURCE_URL_LOCAL = os.environ.get("RESOURCE_URL_LOCAL", "ssh://localhost")
WORKING_DIRECTORY = os.environ.get("WORKING_DIRECTORY", os.path.join(os.environ.get("HOME", "/tmp"), "work"))

if __name__ == "__main__":

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    run_mini_app_benchmark()
