import os

import dask

try:
    from dask_cuda import LocalCUDACluster
except:
    print("Failed to import dask cuda..")

from distributed import LocalCluster, Client, wait

from cluster.base_executor import Executor


def initialize_client(cluster_config):
    if "address" in cluster_config:
        # Connect to an existing remote cluster
        address = cluster_config["address"]
        client = Client(address)
    elif "scheduler_file" in cluster_config:
        # Connect to an existing remote cluster
        client = Client(scheduler_file=cluster_config["scheduler_file"])
    else:
        cluster_type = cluster_config.get("type", "local")

        if cluster_type == "local":
            cluster = LocalCluster(**cluster_config.get("local", {}))
        elif cluster_type == "local-cuda":
            cluster = LocalCUDACluster(**cluster_config.get("local", {}))
        else:
            raise ValueError(f"Unsupported cluster type: {cluster_type}")

        client = Client(cluster)

    print(client.scheduler_info())
    return client


class DaskExecutor(Executor):
    def __init__(self, cluster_config=None):
        super().__init__()
        self.cluster_config = cluster_config or {}
        self.client = initialize_client(self.cluster_config["config"])

    def close(self):
        self.client.close()

    def get_client(self):
        return self.client

    def task(self, func):
        def wrapper(*args, **kwargs):
            return self.client.submit(func, *args, **kwargs)

        return wrapper

    def run_sync_task(self, func, *args, **kwargs):
        print(f"Running qtask with args {args}, kwargs {kwargs}")
        wrapper_func = self.task(func)
        return wrapper_func(*args, **kwargs).result()

    @staticmethod
    def get_command(**kwargs):
        args_str = " ".join(kwargs['args'])
        working_directory = kwargs['working_directory']
        output_file = kwargs['output_file']
        cmd = f"cd {working_directory} && {kwargs['executable']} {args_str}  > {output_file} 2>&1"
        return cmd

    def submit_async_process(self, **kwargs):
        print(f"Running task with cmd {kwargs}")
        run_func = dask.delayed(os.system)
        cmd = self.get_command(**kwargs)
        return self.client.compute(run_func(cmd), resources=kwargs['resources'])

    def submit_tasks(self, input_tasks, compute_func, *args):
        result_bag = input_tasks.map(lambda x: compute_func(x, *args))
        return self.client.compute(result_bag)

    @staticmethod
    def wait(futures):
        wait(futures)
