from cluster.DaskExecutor import DaskExecutor


class MiniAppExecutor:
    def __init__(self, cluster_config):
        self.cluster_config = cluster_config

    def get_executor(self):
        executor_type = self.cluster_config.get("executor", "dask")

        if executor_type == "dask":
            return DaskExecutor(self.cluster_config)
        # elif executor_type == "ray":
        #     return RayExecutor(self.cluster_config).get_client()
        else:
            raise ValueError(f"Unsupported cluster type: {executor_type}")


