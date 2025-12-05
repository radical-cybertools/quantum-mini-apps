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
from pilot.dreamer import DreamerStrategyType

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.circuit_cutting_dreamer.motif import (
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
    DREAMER_STRATEGY,
    CircuitCuttingBuilder
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define benchmark configurations at the top
BENCHMARK_CONFIG = {
    'num_runs': 1,
    'num_pilots': 1,  # Number of pilots to create
    'hardware_configs': [
        {
            'nodes': [1],
            'cores_per_node': 16,
            'gpus_per_node': [0]
        }
    ],
    'circuit_configs': [
        {
            'qubit_sizes': [34],
            'subcircuit_sizes': [6, 4, 2],  # 30//4 + 1
            'num_samples': 5000
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
            .set_dreamer_strategy(self.parameters[DREAMER_STRATEGY])
            .build(self.executor)
        )

        cc.run()

    def close(self):
        self.executor.close()




def create_cluster_info(nodes, cores, gpus, num_pilots=2):
    """
    Create cluster configuration with programmable number of pilots.
    
    Args:
        nodes: Number of nodes per pilot
        cores: Number of cores per node
        gpus: Number of GPUs per node
        num_pilots: Number of pilots to create (default: 2)
    
    Returns:
        Dictionary containing pilot configuration
    """
    # Define the base pilot template
    base_pilot_template = {
        "resource": RESOURCE_URL_LOCAL,
        "working_directory": WORKING_DIRECTORY,
        "type": "ray",
        "number_of_nodes": nodes,
        "cores_per_node": int(cores),
        "gpus_per_node": gpus,
        "QPUs": 1,
        "resource_type": "quantum",
        "quantum": {
            "executor": "qiskit_local",  # Uses framework-provided Qiskit backends
            # No custom_backends - will use framework defaults
            "backend_options": {
                "shots": 4096,
                "device": "CPU",
                "method": "statevector"
            }                
        }
    }
    
    # Generate pilots list by replicating the template
    pilots = []
    for i in range(num_pilots):
        pilot = base_pilot_template.copy()
        pilot["name"] = f"pilot_{i+1}"  # Add unique name for each pilot
        pilots.append(pilot)
    
    return {
        "executor": "pilot",
        "config": {
            "type": "ray",
            "working_directory": WORKING_DIRECTORY,
            "dreamer_strategy": DreamerStrategyType.ROUND_ROBIN,
            "pilots": pilots
        }
    }

def create_cluster_info_with_custom_pilots(nodes, cores, gpus, pilot_configs):
    """
    Create cluster configuration with custom pilot configurations.
    
    Args:
        nodes: Number of nodes per pilot
        cores: Number of cores per node
        gpus: Number of GPUs per node
        pilot_configs: List of custom pilot configurations
    
    Returns:
        Dictionary containing pilot configuration
    """
    pilots = []
    for i, pilot_config in enumerate(pilot_configs):
        pilot = {
            "resource": RESOURCE_URL_LOCAL,
            "working_directory": WORKING_DIRECTORY,
            "type": "ray",
            "number_of_nodes": nodes,
            "cores_per_node": cores,
            "gpus_per_node": gpus,
            "resource_type": "quantum",
            "name": f"pilot_{i+1}",
            "quantum": {
                "executor": "qiskit_local",
                "backend_options": pilot_config.get("backend_options", {
                    "shots": 4096,
                    "device": "CPU",
                    "method": "statevector"
                })
            }
        }
        pilots.append(pilot)
    
    return {
        "executor": "pilot",
        "config": {
            "type": "ray",
            "working_directory": WORKING_DIRECTORY,
            "dreamer_strategy": DreamerStrategyType.ROUND_ROBIN,
            "pilots": pilots
        }
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
    return {
        SUBCIRCUIT_SIZE: subcircuit_size,
        BASE_QUBITS: circuit_size,
        SCALE_FACTOR: 1,
        OBSERVABLES: ["Z" + "I" * (circuit_size - 1)],
        NUM_SAMPLES: num_samples,
        SUB_CIRCUIT_TASK_RESOURCES: {
            "num_cpus": 1,
            "num_gpus": 0,  # No GPU needed for CPU simulation
            "QPU": 1,
        },
        FULL_CIRCUIT_TASK_RESOURCES: {
            "num_cpus": 16,
            "num_gpus": num_gpus,
            "num_nodes": num_nodes,
            "memory": None,
        },
        FULL_CIRCUIT_ONLY: True,
        CIRCUIT_CUTTING_ONLY: False,
        CIRCUIT_CUTTING_SIMULATOR_BACKEND_OPTIONS: {
            "backend_options": {"shots": 4096, "device":"CPU", "method":"statevector"},
            # "backend_options": {"device":"GPU", "method":"statevector", "shots": 4096,
            #                   "blocking_enable":True, "batched_shots_gpu":True, 
            #                   "blocking_qubits":23},
            "mpi": False
        },
        DREAMER_STRATEGY: DreamerStrategyType.ROUND_ROBIN,
        FULL_CIRCUIT_SIMULATOR_BACKEND_OPTIONS: {
            "backend_options": {"shots": 4096, "device":"CPU", "method":"statevector"},
            # "backend_options": {"device":"GPU", "method":"statevector", "shots": 4096,
            #                   "blocking_enable":True, "batched_shots_gpu":True, 
            #                   "blocking_qubits":23},
            "mpi": False
        },
        SCENARIO_LABEL: f"circuit_size_{circuit_size}_subcircuit_{subcircuit_size}_samples_{num_samples}_cores_{num_cores}_nvidia_80GB"
    }

def run_mini_app_benchmark():
    for run_idx in range(BENCHMARK_CONFIG['num_runs']):
        logger.info(f"Starting benchmark run {run_idx + 1}/{BENCHMARK_CONFIG['num_runs']}")
        
        for num_pilots in range(1, BENCHMARK_CONFIG['num_pilots'] + 1):
            for hw_config in BENCHMARK_CONFIG['hardware_configs']:
                for nodes in hw_config['nodes']:
                    for gpus in hw_config['gpus_per_node']:
                        cluster_info = create_cluster_info(
                            nodes=nodes,
                            cores=hw_config['cores_per_node'],
                            gpus=gpus,
                            num_pilots=num_pilots
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

if __name__ == "__main__":
    RESOURCE_URL_HPC = "slurm://localhost"
    RESOURCE_URL_LOCAL = "ssh://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    run_mini_app_benchmark()
