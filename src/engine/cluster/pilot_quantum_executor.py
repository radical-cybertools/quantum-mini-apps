import os, dask, ray
from pilot.pilot_compute_service import PilotComputeService
from pilot.pilot_enums_exceptions import ExecutionEngine
from distributed import Client, wait
import dask.bag as db
from engine.cluster.base_executor import Executor
from engine.cluster.dask_executor import DaskExecutor   


def initialize_client(cluster_config):
    pilot_compute_description = cluster_config
    execution_engine = ExecutionEngine(cluster_config["type"])
    working_directory = cluster_config["working_directory"]
    
    pcs = PilotComputeService(execution_engine=execution_engine, working_directory=working_directory)
    pilot = pcs.create_pilot(pilot_compute_description=pilot_compute_description)
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
        self.pilot.cancel()
        

    def submit_tasks(self, compute_func, *args, **kwargs):
        if self.type == "dask":
            return self.submit_tasks_dask(compute_func, *args,  **kwargs)
        elif self.type == "ray":
            return self.submit_tasks_ray(compute_func, *args,  **kwargs)
        
    def submit_task(self, compute_func, *args, **kwargs):
        return self.pilot.submit_task(compute_func, *args, **kwargs)
    
    def submit_mpi_task(self, *args, **kwargs):
        return self.pilot.submit_mpi_task(*args, **kwargs)    

    
    def submit_tasks_dask(self, compute_func, *args,  **kwargs):
        circuits_observables = args[0]
        circuit_bag = db.from_sequence(circuits_observables)
        args = args[1:]
        return circuit_bag.map(lambda x: self.pilot.submit_task(compute_func, x, *args, **kwargs))


    def submit_tasks_ray(self, compute_func, *args,  **kwargs):
        input_tasks = args[0]  # The first argument is the collection of tasks
        args = args[1:]  # Remove the first argument
        
        return [self.pilot.submit_task(compute_func, task, *args,  **kwargs) for task in input_tasks]


    def wait(self, futures):
        self.pilot.wait_tasks(futures)
        
    def get_results(self, futures):
        return self.pilot.get_results(futures)

