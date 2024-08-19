import os, dask
import pennylane as qml
from pilot.pilot_compute_service import PilotComputeService
from distributed import Client, wait

from engine.cluster.base_executor import Executor
from engine.cluster.dask_executor import DaskExecutor   


def initialize_client(cluster_config):
    pilot_compute_description_dask = cluster_config

    pcs = PilotComputeService()
    pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    dask_client = pcs.get_client()
    return dask_client


class PilotQuantumExecutor(Executor):
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
