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


from engine.manager import MiniAppExecutor

from utils.discrete_qcbm_model_handler import (
    DiscreteQCBMModelHandler,
)
from qugen.main.data.data_handler import load_data
import os


class QMLTraining:
    def __init__(self, cluster_config, parameters=None):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters

    def run(self):        
        data_set_name = "X_2D"

        # Construct the path to the dataset within the subpackage
        package_path = os.path.dirname(__file__)
        data_set_path = os.path.join(package_path, "data", data_set_name)

        data, _ = load_data(data_set_path)
        model = DiscreteQCBMModelHandler()

        # build a new model:

        model.build(
            "discrete",
            data_set_name,
            n_qubits=8,
            n_registers=2,
            circuit_depth=2,
            initial_sigma=0.01,
            circuit_type="copula",
            transformation="pit",
            hot_start_path="", #path to pre-trained model parameters
        )

        # train a quantum generative model:

        model.train(
            data,
            n_epochs=500,
            batch_size=200,
            hist_samples=100000,
        )

        # evaluate the performance of the trained model:

        evaluation_df = model.evaluate(data)

        # find the model with the minimum Kullbach-Liebler divergence:

        minimum_kl_data = evaluation_df.loc[evaluation_df["kl_original_space"].idxmin()]
        minimum_kl_calculated = minimum_kl_data["kl_original_space"]
        print(f"{minimum_kl_calculated=}")



if __name__ == "__main__":
    cluster_info = {
        "executor": "dask",
        "config": {
            "type": "local",
            "local": {
                "n_workers": 4,
                "threads_per_worker": 2,
                "memory_limit": "4GB"
            }
        }

    }

    ce_parameters = {
        QUBITS: 10,
        NUM_ENTRIES: 10,
        CIRCUIT_DEPTH: 1,
        SIZE_OF_OBSERVABLE: 1
    }


    qs = QMLTraining(cluster_info, ce_parameters)
    qs.run()










# # --------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------
# # ADDITIONAL FUNCTIONALITY
# # Uncomment the following section if you want to train a model for already pre-trained weights

# # load existing model:

# model_name  = 'discrete_X_2D_copula_pit_qcbm_d320'  # example model name
# new_model = DiscreteQCBMModelHandler().reload(model_name, epoch=500)

# # re-train model from pre-trained existing model

# new_model.train(data,
#             n_epochs = 25,
#             batch_size=200,
#             hist_samples = 10000,
#             )

# # generate samples from a trained model:

# number_samples = 10000
# samples = new_model.predict(number_samples)

# # plot 2D samples:

# import matplotlib.pyplot as plt
# plt.scatter(samples[:, 0], samples[:, 1])
# plt.savefig('generated_samples.png')
