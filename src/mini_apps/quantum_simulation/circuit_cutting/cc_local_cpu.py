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
from mini_apps.quantum_simulation.motifs.circuit_cutting_motif import (
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


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def kill_processes_by_keyword(keyword):
    """
    Kills all processes whose command line contains the specified keyword.
    """
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            # Check if the process's command line contains the keyword
            if proc.info["cmdline"] and any(
                keyword in arg for arg in proc.info["cmdline"]
            ):
                pid = proc.info["pid"]
                print(
                    f"Killing process {pid} ({proc.info['name']}) with command: {proc.info['cmdline']}"
                )
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process may have terminated or we don't have permissions
            pass


def stop_ray():
    try:
        # Execute the "ray stop" command
        result = subprocess.run(
            ["ray", "stop"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("Ray stopped successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error stopping Ray:")
        print(e.stderr)


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
        kill_processes_by_keyword("pilot.plugins.ray_v2.agent")
        stop_ray()


# Define benchmark configurations at the top
BENCHMARK_CONFIG = {
    'num_runs': 3,
    'hardware_configs': [
        {
            'nodes': 1,
            'cores_per_node': 128,
            'gpus_per_node': [1, 2, 4]
        }
    ],
    'circuit_configs': [
        {
            'qubit_sizes': [33],
            'subcircuit_sizes': [8],  # 30//4 + 1
            'num_samples': 100
        }
    ]
}

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

def create_cluster_info_perlmutter(nodes, cores, gpus):
    return {       
        "executor": "pilot",
        "config": {
            "resource": RESOURCE_URL_HPC,
            "working_directory": WORKING_DIRECTORY,
            "type": "ray",
            "number_of_nodes": nodes,
                "cores_per_node": 64,
                "gpus_per_node": 4,
                "queue": "premium",
                "walltime": 120,            
                "project": "m4408",
                "scheduler_script_commands": ["#SBATCH --constraint=gpu&hbm80g",
                                                "#SBATCH --gpus-per-task=1",
                                                "#SBATCH --ntasks-per-node=4",
                                                "#SBATCH --gpu-bind=none"],
            }
        }

def create_cc_parameters(circuit_size, subcircuit_size, num_samples, num_cores, num_gpus):
    return {
        SUBCIRCUIT_SIZE: subcircuit_size,
        BASE_QUBITS: circuit_size,
        SCALE_FACTOR: 1,
        OBSERVABLES: ["Z" + "I" * (circuit_size - 1)],
        NUM_SAMPLES: num_samples,
        SUB_CIRCUIT_TASK_RESOURCES: {
            "num_cpus": 1,
            "num_gpus": 0,
            "memory": None,
        },
        FULL_CIRCUIT_TASK_RESOURCES: {
            "num_cpus": 1,
            "num_gpus": num_gpus,
            "num_nodes": 1,
            "memory": None,
        },
        FULL_CIRCUIT_ONLY: True,
        CIRCUIT_CUTTING_ONLY: False,
        CIRCUIT_CUTTING_SIMULATOR_BACKEND_OPTIONS: {
            "backend_options": {"shots": 1024, "device":"CPU", "method":"statevector"},
            "mpi": False
        },
        FULL_CIRCUIT_SIMULATOR_BACKEND_OPTIONS: {
            "backend_options": {"device":"GPU", "method":"statevector", 
                              "blocking_enable":True, "batched_shots_gpu":True, 
                              "blocking_qubits":23},
            "mpi": True
        },
        SCENARIO_LABEL: f"circuit_size_{circuit_size}_subcircuit_{subcircuit_size}_samples_{num_samples}_cores_{num_cores}_nvidia_80GB"
    }

def run_mini_app_benchmark():
    for run_idx in range(BENCHMARK_CONFIG['num_runs']):
        logger.info(f"Starting benchmark run {run_idx + 1}/{BENCHMARK_CONFIG['num_runs']}")
        
        for hw_config in BENCHMARK_CONFIG['hardware_configs']:
            for gpus in hw_config['gpus_per_node']:
                cluster_info = create_cluster_info_perlmutter(
                    hw_config['nodes'], 
                    hw_config['cores_per_node'], 
                    gpus
                )
                
                for circuit_config in BENCHMARK_CONFIG['circuit_configs']:
                    for qubit_size in circuit_config['qubit_sizes']:
                        for subcircuit_size in circuit_config['subcircuit_sizes']:
                            try:
                                cc_parameters = create_cc_parameters(
                                    qubit_size,
                                    subcircuit_size,
                                    circuit_config['num_samples'],
                                    hw_config['cores_per_node'],
                                    gpus
                                )
                                
                                logger.info(f"Running configuration: {cc_parameters[SCENARIO_LABEL]}")
                                qs = QuantumSimulation(cluster_info, cc_parameters)
                                qs.run()
                                qs.close()
                                
                            except Exception as e:
                                logger.error(f"Error in configuration {cc_parameters[SCENARIO_LABEL]}: {e}")
                                raise e

if __name__ == "__main__":
    RESOURCE_URL_HPC = "slurm://localhost"
    RESOURCE_URL_LOCAL = "ssh://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    run_mini_app_benchmark()
