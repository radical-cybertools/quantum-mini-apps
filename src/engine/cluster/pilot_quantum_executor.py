import os, dask, ray
import pennylane as qml
from pilot.pilot_compute_service import PilotComputeService
from distributed import Client, wait
import dask.bag as db
from engine.cluster.base_executor import Executor
from engine.cluster.dask_executor import DaskExecutor   


def initialize_client(cluster_config):
    pilot_compute_description_dask = cluster_config
    pcs = PilotComputeService()
    pilot = pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    pilot.wait()
    dask_ray_client = pilot.get_client()
    return pilot, dask_ray_client


class PilotQuantumExecutor(Executor):

    def __init__(self, cluster_config=None):
        super().__init__()
        self.cluster_config = cluster_config or {}
        self.type = self.cluster_config["config"]["type"]
        self.pilot, self.client = initialize_client(self.cluster_config["config"])

    def close(self):
        # self.client.close()
        self.pilot.cancel()
        

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

    def submit_tasks(self, compute_func, *args):
        if self.type == "dask":
            return self.submit_tasks_dask(compute_func, *args)
        elif self.type == "ray":
            return self.submit_tasks_ray(compute_func, *args)

    
    def submit_tasks_dask(self, compute_func, *args):
        circuits_observables = args[0]
        circuit_bag = db.from_sequence(circuits_observables)
        args = args[1:]
        result_bag = circuit_bag.map(lambda x: compute_func(x, *args))
        return self.client.compute(result_bag)


    def submit_tasks_ray(self, compute_func, *args):
        input_tasks = args[0]  # The first argument is the collection of tasks
        args = args[1:]  # Remove the first argument

        # Define a remote function to compute each task
        @ray.remote
        def compute_remote(x, *args):
            return compute_func(x, *args)

        with self.client as ray_client:
            # Map the remote function across the input tasks using Ray
            result_futures = [compute_remote.remote(task, *args) for task in input_tasks]

        return result_futures


    def wait(self, futures):
        if self.type == "dask":
            return wait(futures)
        elif self.type == "ray":
            with self.client as ray_client:
                return ray.get(futures)
