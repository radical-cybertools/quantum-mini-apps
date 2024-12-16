import os
import sys
import math
import datetime
import time
import logging
import psutil
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.motifs.circuit_cutting_motif import (
    BASE_QUBITS,
    NUM_SAMPLES,
    OBSERVABLES,
    SCALE_FACTOR,
    SIMULATOR_BACKEND_OPTIONS,
    SUB_CIRCUIT_TASK_RESOURCES,
    SUBCIRCUIT_SIZE,
    FULL_CIRCUIT_TASK_RESOURCES,
    FULL_CIRCUIT_ONLY,
    CIRCUIT_CUTTING_ONLY,
    CircuitCuttingBuilder
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# import pdb


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
            .set_num_samples(self.parameters[NUM_SAMPLES])
            .build(self.executor)
        )

        # pdb.set_trace()
        cc.run()

    def close(self):
        self.executor.close()
        # hack terminate all agents
        kill_processes_by_keyword("pilot.plugins.ray_v2.agent")
        stop_ray()


if __name__ == "__main__":
    RESOURCE_URL_HPC = "ssh://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    circuit_sizes = [33]
    subcircuit_sizes = {
        circuit_size: [circuit_size // 4 + 1]
        for circuit_size in circuit_sizes
    }

    # subcircuit_sizes = {
    #     circuit_size: [size for size in range(2, circuit_size // 2 + 1, 2)]
    #     for circuit_size in circuit_sizes
    # }
    # subcircuit_sizes = {8: [4]}

    for num_cores_per_node in [128]:
        for circuit_size in circuit_sizes:
            for subcircuit_size in subcircuit_sizes[circuit_size]:
                for num_samples in [100]:
                    try:
                        cluster_info = {
                            "executor": "pilot",
                            "config": {
                                "resource": RESOURCE_URL_HPC,
                                "working_directory": WORKING_DIRECTORY,
                                "type": "ray",
                                "number_of_nodes": 1,
                                "cores_per_node": num_cores_per_node,
                                "gpus_per_node": 0,
                            },
                        }

                        cc_parameters = {
                            SUBCIRCUIT_SIZE: subcircuit_size,
                            BASE_QUBITS: circuit_size,
                            SCALE_FACTOR: 1,
                            OBSERVABLES: [
                                "Z" + "I" * (circuit_size - 1)
                            ],  # ["ZIIIIII", "IIIZIII", "IIIIIII"],
                            NUM_SAMPLES: num_samples,
                            SUB_CIRCUIT_TASK_RESOURCES: {
                                "num_cpus": 1,
                                "num_gpus": 0,
                                "memory": None,
                            },
                            FULL_CIRCUIT_TASK_RESOURCES: {
                                "num_cpus": 1,
                                "num_gpus": 0,
                                "memory": None,
                            },
                            FULL_CIRCUIT_ONLY: True,
                            CIRCUIT_CUTTING_ONLY: False
                            # SIMULATOR_BACKEND_OPTIONS: {"backend_options": {"shots": 4096, "device":"GPU", "method":"statevector", "blocking_enable":True, "batched_shots_gpu":True, "blocking_qubits":25}}
                        }

                        logger.info(
                            f"******* Running simulation with configuration: cluster_info={cluster_info}, cc_parameters={cc_parameters}"
                        )

                        qs = QuantumSimulation(cluster_info, cc_parameters)
                        qs.run()
                        logger.debug("Stop Executor")
                        qs.close()
                        # time.sleep(60)
                    except Exception as e:
                        print(f"Error: {e}")
                        raise e
