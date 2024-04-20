import os

from miniappengine.manager import MiniAppExecutor
from miniappengine.motifs.dist_state_vector_motif import DistStateVector


class QuantumSimulation:
    def __init__(self, cluster_config, parameters=None):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters

    def run(self):
        sv = DistStateVector(self.executor, **self.parameters)
        sv.run()


if __name__ == "__main__":
    scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler_file.json")
    cluster_info = {
        "executor": "dask",
        "config": {
            "scheduler_file": scheduler_file
        }
    }

    sv_parameters = {
        "num_runs": 2,
        "n_wires": 10,
        "n_layers": 2,
        "diff_method": "adjoint",
        "pennylane_device_config": {
            "name": 'lightning.qubit',
            "wires": 10,
        }
    }

    qs = QuantumSimulation(cluster_info, sv_parameters)
    qs.run()
