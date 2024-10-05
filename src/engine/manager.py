from engine.cluster.dask_executor import DaskExecutor
from engine.cluster.pilot_quantum_executor import PilotQuantumExecutor
from engine.cluster.multi_pilot_quantum_executor import MultiPilotQuantumExecutor


class MiniAppExecutor:
    def __init__(self, cluster_config):
        self.cluster_config = cluster_config

    def get_executor(self):
        executor_type = self.cluster_config.get("executor", "dask")
        # executor 
        if executor_type == "pilot":
            return PilotQuantumExecutor(self.cluster_config)
        if executor_type == "multi-pilot":
            return MultiPilotQuantumExecutor(self.cluster_config)
        if executor_type == "dask":
            return DaskExecutor(self.cluster_config)
        # elif executor_type == "ray":
        #     return RayExecutor(self.cluster_config).get_client()
        else:
            raise ValueError(f"Unsupported cluster type: {executor_type}")


