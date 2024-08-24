# This file is part of the Quantum Mini-Apps project, based on original work 
# adapted from other open-source projects. Contributions made to this file 
# are licensed under the terms of the Apache License, Version 2.0.

# QuGEN Copyright notice: https://github.com/QutacQuantum/qugen
# Copyright 2023 QUTAC, BASF Digital Solutions GmbH, BMW Group, 
# Lufthansa Industry Solutions AS GmbH, Merck KGaA (Darmstadt, Germany), 
# Munich Re, SAP SE.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# system imports
import os
import time

# MiniApp framework imports
from engine.manager import MiniAppExecutor
from engine.metrics.csv_writer import MetricsFileWriter

# QuGEN imports
from utils.discrete_qcbm_model_handler import (
    DiscreteQCBMModelHandler,
)
from qugen.main.data.data_handler import load_data





class QMLTrainingMiniApp:

    def __init__(self, cluster_config, parameters=None, scenario_label="QML Training MiniApp"):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters
        self.scenario_label = scenario_label

        self.model = None
        self.file_name = f"qml_result_{self.current_datetime.strftime('%Y%m%d_%H%M%S')}.csv"
        self.result_file = os.path.join(self.result_dir, self.file_name)
        header = ["timestamp", "scenario_label", "num_qubits",  "compute_time_sec", "parameters", "cluster_info"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)


    def run_training(self):                
        # Construct the path to the dataset within the subpackage
        package_path = os.path.dirname(__file__)
        data_set_path = os.path.join(package_path, "data", self.parameters["build_parameters"]["data_set_name"])

        data, _ = load_data(data_set_path)
        self.model = DiscreteQCBMModelHandler()

        # build a new model:
        self.model.build(
            self.parameters["build_parameters"]['model_type'],
            self.parameters["build_parameters"]['data_set_name'],
            n_qubits=self.parameters["build_parameters"]['n_qubits'],
            n_registers=self.parameters["build_parameters"]['n_registers'],
            circuit_depth=self.parameters["build_parameters"]['circuit_depth'],
            initial_sigma=self.parameters["build_parameters"]['initial_sigma'],
            circuit_type=self.parameters["build_parameters"]['circuit_type'],
            transformation=self.parameters["build_parameters"]['transformation'],
            hot_start_path=self.parameters["build_parameters"]['hot_start_path'],  # path to pre-trained model parameters
            parallelism_framework = self.parameters["build_parameters"]['parallelism_framework']
        )

        # train a quantum generative model:
        self.model.train(
            data,  # Assuming 'data' is still an external variable not in the parameters dictionary
            n_epochs=self.parameters["train_parameters"]['n_epochs'],
            batch_size=self.parameters["train_parameters"]['batch_size'],
            hist_samples=self.parameters["train_parameters"]['hist_samples'],
        )
        
        # evaluate the performance of the trained model:
        evaluation_df = self.model.evaluate(data)

        # find the model with the minimum Kullbach-Liebler divergence:
        minimum_kl_data = evaluation_df.loc[evaluation_df["kl_original_space"].idxmin()]
        minimum_kl_calculated = minimum_kl_data["kl_original_space"]
        print(f"{minimum_kl_calculated=}")


    def run(self): 
         # Submit all the tasks
        futures = self.executor.submit_tasks(self.run_training)

        # wait for the tasks to complete
        start_time = time.time()
        self.executor.wait(futures)
        end_time = time.time()
        compute_time_ms = end_time-start_time
        self.metrics_file_writer.write([self.timestamp, self.scenario_label, self.num_qubits, 
                                        compute_time_ms, str(self.parameters), str(self.cluster_info)])

        self.metrics_file_writer.close()


if __name__ == "__main__":
    RESOURCE_URL_HPC = "slurm://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")
    CORES_PER_NODE = 2

    cluster_info = {       
                    "executor": "pilot",
                    "config": {
                        "resource": RESOURCE_URL_HPC,
                        "working_directory": WORKING_DIRECTORY,
                        "type": "ray",
                        "number_of_nodes": 1,
                        "cores_per_node": CORES_PER_NODE,
                        "gpus_per_node": 0,
                        "queue": "debug",
                        "walltime": 30,            
                        "project": "m4408",
                        "conda_environment": "/pscratch/sd/l/luckow/conda/quantum-mini-apps2",
                        "scheduler_script_commands": ["#SBATCH --constraint=cpu"]
                    }
                }
    
    qml_parameters = {
        "build_parameters": {
            'model_type': "discrete",
            'data_set_name': "X_2D",
            'n_qubits': 8,
            'n_registers': 2,
            'circuit_depth': 2,
            'initial_sigma': 0.01,
            'circuit_type': "copula",
            'transformation': "pit",
            'hot_start_path': "",  # path to pre-trained model parameters
            "parallelism_framework": "jax"
        },
        "train_parameters": {            
            'n_epochs': 3,
            'batch_size': 200,
            'hist_samples': 100000
        }
    }

    qml_mini_app = QMLTrainingMiniApp(cluster_info, qml_parameters)
    #qml_mini_app.update_parameters(qml_parameters)
    qml_mini_app.run()

    qml_mini_app.close()








