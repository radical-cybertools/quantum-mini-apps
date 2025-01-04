import os
from datetime import datetime
from time import perf_counter
from pennylane import numpy as np
import yaml
from math import ceil

from engine.metrics.csv_writer import MetricsFileWriter
from engine.manager import MiniAppExecutor

from mini_apps.qml_data_compression.utils.sweeping import staircase_circuit
from mini_apps.qml_data_compression.utils.state_encoding import FRQI_RGBa_encoding, calc_MPS, right_canonical
from mini_apps.qml_data_compression.utils.unitary_to_pennylane import UnitaryToPennylane

from time import perf_counter

import argparse


parser = argparse.ArgumentParser(description="QML Data Compression Mini App")
parser.add_argument("--num_nodes", type=int, required=True, help="Number of nodes to use for computation")
arg = parser.parse_args()

class QMLCompressionMiniApp:
    def __init__(self, pilot_compute_description, app_config):
        self.app_config = app_config

        os.makedirs(pilot_compute_description["config"]["working_directory"], exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"result_compression_{current_time}.csv"
        self.result_file = os.path.join(pilot_compute_description["config"]["working_directory"], file_name)
        print(f"Result file: {self.result_file}")
        header = ["n_samples", "number_of_nodes", "cores_per_node", "compute_time_sec"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)
        self.executor = MiniAppExecutor(pilot_compute_description).get_executor()

    def run(self, pilot_compute_description):
        start = perf_counter()
        # batch_size = ceil(self.app_config["n_samples"] / pilot_compute_description["number_of_nodes"] / pilot_compute_description["cores_per_node"])
        # batched_indices = np.arange(self.app_config["n_samples"]).reshape(-1, batch_size)
        processes_max = pilot_compute_description["number_of_nodes"] * pilot_compute_description["cores_per_node"]
        n_batches = next(processes_max - i for i in range(processes_max) if self.app_config["n_samples"] % (processes_max - i) == 0)
        batched_indices = np.arange(self.app_config["n_samples"]).reshape(n_batches, -1)
        batched_indices = [indices.tolist() for indices in batched_indices]
        futures = self.executor.submit_tasks(workflow_batch, batched_indices)
        self.executor.wait(futures)
        compute_time_sec = perf_counter() - start
        print(f"Completed all images for {num_cpus} CPUs")
        self.metrics_file_writer.write([
            self.app_config["n_samples"],
            pilot_compute_description["number_of_nodes"],
            pilot_compute_description["cores_per_node"],
            compute_time_sec])

def workflow(index: int):
    # Step 0: Encode the image the image as quantum state and transform it to target MPS
    times = {}
    times["time_start_loading"] = perf_counter()
    image = np.load(f"/pscratch/sd/f/fkiwit/data/{index}.npy", allow_pickle=True)
    states = FRQI_RGBa_encoding(image[None])
    target_tensor, Lambdtarget_tensor = calc_MPS(np.asarray(states))
    target_tensor = right_canonical(target_tensor)
    # Step 1: Fit the staircase circuit to the target MPS
    times["time_sweeping_start"] = perf_counter()
    sc = staircase_circuit(L=int(np.log2(states.size)), layers=4, batchsize=1, orthogonal=False)
    overlaps, time, Bnew = sc.optimize_circuit(target_tensor, iters=100)
    # Step 2: Transform the staircase circuit to a qunantum circuit
    times["time_bfgs_start"] = perf_counter()
    unitary_to_pennylane = UnitaryToPennylane(sc.gates[0])
    circuit, params = unitary_to_pennylane.get_circuit(RY=False)
    # Step 3: Fit the circuit to the target state with BFGS
    result, losses = unitary_to_pennylane.train_circuit(circuit, states, params, use_jit=False, maxiter=50)
    times["time_done"] = perf_counter()

    with open(os.path.join(IMAGE_DIR, f"{index}_times.yml"), "w") as filehandler:
        yaml.dump(times, filehandler)
    np.save(os.path.join(IMAGE_DIR, f"{index}_gates.npy"), sc.gates[0], allow_pickle=True)
    np.save(os.path.join(IMAGE_DIR, f"{index}_params.npy"), result.x, allow_pickle=True)
    print("saved")
    return result

def workflow_batch(indices):
    # Batch the lightweight tasks to maximize CPU utilization
    results = []
    for i, index in enumerate(indices):
        print(i)
        results.append(workflow(index))
    return results

if __name__ == "__main__":
    # TODO: make it more configurable, add number of iterations ..
    app_config = {
        "n_samples": 60000,
        "n_iterations_sweeping": None,
        "n_iterations_bfgs": None,
        "n_layers": None,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    WORKING_DIRECTORY = os.path.join(os.environ["PSCRATCH"], f"work_nodes16_32/{timestamp}")
    RESOURCE_URL_HPC = "slurm://localhost"

    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    with open(os.path.join(WORKING_DIRECTORY, "app_config.yml"), "w") as filehandler:
        yaml.dump(app_config, filehandler)


    num_nodes = arg.num_nodes
    num_cpus = 256
    cluster_info = {
        "executor": "pilot",
        "config": {
            "resource": RESOURCE_URL_HPC,
            "working_directory": WORKING_DIRECTORY,
            "number_of_nodes": num_nodes,
            "cores_per_node": num_cpus,
            "gpus_per_node": 0,
            "queue": "premium",
            # "walltime": 30,
            "walltime": int(1920 / num_nodes),
            "type": "ray",
            "project": "m4408",
            "conda_environment": "/pscratch/sd/f/fkiwit/conda/qma/",
            "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
        }
    }

    qml_compression_mini_app = QMLCompressionMiniApp(cluster_info, app_config)
    IMAGE_DIR = os.path.join(WORKING_DIRECTORY, f"data_{num_nodes}_{num_cpus}")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    with open(os.path.join(IMAGE_DIR, "cluster_info.yml"), "w") as filehandler:
        yaml.dump(cluster_info, filehandler)

    qml_compression_mini_app.run(cluster_info["config"])

    qml_compression_mini_app.metrics_file_writer.close()
    qml_compression_mini_app.executor.close()
