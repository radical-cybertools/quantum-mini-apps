from motifs.dist_state_vector_motif import DistStateVector


class QuantumSimulation:
    def __init__(self, cluster_config, parameters=None):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters

    def run(self):
        sv = DistStateVector(self.executor, **self.parameters)
        sv.run()


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
