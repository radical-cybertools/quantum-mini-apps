import os
from datetime import datetime
from time import perf_counter

from engine.metrics.csv_writer import MetricsFileWriter
from engine.manager import MiniAppExecutor

from time import perf_counter, sleep

# from mini_apps.classifier.utils.trainingg import training
from mini_apps.classifier.utils.training import training


# def training(config):
#     sleep(60)
#     print(config)
#     print(f"okok")

class QMLClassifierMiniApp:
    def __init__(self, pilot_compute_description):
        os.makedirs(pilot_compute_description["config"]["working_directory"], exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"result_compression_{current_time}.csv"
        self.result_file = os.path.join(pilot_compute_description["config"]["working_directory"], file_name)
        print(f"Result file: {self.result_file}")
        header = ["column_1", "compute_time_sec"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)
        self.executor = MiniAppExecutor(pilot_compute_description).get_executor()

    def run(self, config):
        start = perf_counter()
        futures = self.executor.submit_tasks(training, [config, config])
        self.executor.wait(futures)
        print("Done")
        compute_time_sec = perf_counter() - start
        self.metrics_file_writer.write([
            "worked",
            compute_time_sec])

if __name__ == "__main__":
    # TODO: make it more configurable, add number of iterations ..

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    WORKING_DIRECTORY = os.path.join(os.environ["PSCRATCH"], f"work_classifier/{timestamp}")
    RESOURCE_URL_HPC = "slurm://localhost"

    os.makedirs(WORKING_DIRECTORY, exist_ok=True)

    cluster_info = {
        "executor": "pilot",
        "config": {
            "resource": RESOURCE_URL_HPC,
            "working_directory": WORKING_DIRECTORY,
            "number_of_nodes": 1,
            "cores_per_node": 256,
            "gpus_per_node": 4,
            "queue": "premium",
            "walltime": 30,
            "type": "ray",
            "project": "m4408",
            "conda_environment": "/pscratch/sd/f/fkiwit/conda/qma/",
            "scheduler_script_commands": ["#SBATCH --constraint=gpu", 
                                          "#SBATCH --gpus-per-task=1",
                                          "#SBATCH --ntasks-per-node=4",
                                          "#SBATCH --gpu-bind=none"],    
        }
    }

    config = {
        "n_qubits": 13,
        "depth": 2,
        "batch_size": 10,
        "n_batches": 20,
        "n_epochs": 1,
        "jit": True,
        "vmap": True,
        "device": "gpu"
    }

    qml_compression_mini_app = QMLClassifierMiniApp(cluster_info)
    qml_compression_mini_app.run(config)

    qml_compression_mini_app.metrics_file_writer.close()
    qml_compression_mini_app.executor.close()
