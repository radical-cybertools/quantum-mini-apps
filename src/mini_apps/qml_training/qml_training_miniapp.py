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
import datetime
import os
import time

# MiniApp framework imports
from engine.manager import MiniAppExecutor
from engine.metrics.csv_writer import MetricsFileWriter

# QuGEN imports
from mini_apps.qml_training.utils.discrete_qcbm_model_handler import (
    DiscreteQCBMModelHandler,
)
from qugen.main.data.data_handler import load_data


class QMLTrainingMiniApp:

    def __init__(self, cluster_config, parameters=None, scenario_label="QML Training MiniApp"):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters
        self.scenario_label = scenario_label
        self.cluster_config = cluster_config
        self.model = None
        self.current_datetime = datetime.datetime.now()
        self.timestamp = self.current_datetime.strftime('%Y-%m-%dT%H:%M:%S')
        self.file_name = f"qml_result_{self.timestamp}.csv"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        # check whether dir exists if not create it
        self.result_dir = os.path.join(script_dir, "results")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.result_file = os.path.join(self.result_dir, self.file_name)
        header = ["timestamp", "scenario_label", "num_qubits",  "compute_time_sec", "parameters", "cluster_info"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)

    def run(self): 
        start_time = time.time()
        # Submit the standalone function instead of the instance method
        futures = self.executor.submit_task(run_training_task, self.parameters)
        result = self.executor.get_results([futures])[0]
        end_time = time.time()
        compute_time_ms = end_time - start_time
        
        self.metrics_file_writer.write([
            self.timestamp,
            self.scenario_label, 
            self.parameters["build_parameters"]['n_qubits'],
            compute_time_ms,
            str(self.parameters),
            str(self.cluster_config)
        ])
        
        self.metrics_file_writer.close()

    def close(self): 
        self.executor.close()


def run_training_task(parameters):
    try:
        # Create a new instance for each task
        model = DiscreteQCBMModelHandler()
        
        # Construct the path to the dataset
        package_path = os.path.dirname(os.path.abspath(__file__))
        data_set_path = os.path.join(package_path, "data", parameters["build_parameters"]["data_set_name"])
        
        data, _ = load_data(data_set_path)
        
        # Build and train model
        model.build(
            parameters["build_parameters"]['model_type'],
            parameters["build_parameters"]['data_set_name'],
            n_qubits=parameters["build_parameters"]['n_qubits'],
            n_registers=parameters["build_parameters"]['n_registers'],
            circuit_depth=parameters["build_parameters"]['circuit_depth'],
            circuit_type=parameters["build_parameters"]['circuit_type'],
            transformation=parameters["build_parameters"]['transformation'],
            hot_start_path=parameters.get("build_parameters", {}).get('hot_start_path', ''),
            parallelism_framework=parameters["build_parameters"]['parallelism_framework']
        )
        
        model.train(
            data,
            n_epochs=parameters["train_parameters"]['n_epochs'],
            batch_size=parameters["train_parameters"]['batch_size'],
            hist_samples=parameters["train_parameters"]['hist_samples'],
        )
        
        evaluation_df = model.evaluate(data)
        minimum_kl_data = evaluation_df.loc[evaluation_df["kl_original_space"].idxmin()]
        return minimum_kl_data["kl_original_space"]
    except Exception as e:
        print(f"Error in run_training_task: {str(e)}")
        raise


if __name__ == "__main__":
    RESOURCE_URL_HPC = "ssh://localhost"
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
                        "conda_environment": "/pscratch/sd/l/luckow/conda/quantum-mini-apps-qml",
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
            'n_epochs': 1,
            'batch_size': 200,
            'hist_samples': 100000
        }
    }

    try:

        qml_mini_app = QMLTrainingMiniApp(cluster_info, qml_parameters)
        #qml_mini_app.update_parameters(qml_parameters)
        qml_mini_app.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        qml_mini_app.close()


