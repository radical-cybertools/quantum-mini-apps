import os, dask, ray
import random
import pennylane as qml
from pilot.pilot_compute_service import PilotComputeService
from pilot.pilot_enums_exceptions import ExecutionEngine
from distributed import Client, wait
import dask.bag as db
from engine.cluster.base_executor import Executor
from engine.cluster.dask_executor import DaskExecutor   


def initialize_client(cluster_config):
    execution_engine = ExecutionEngine(cluster_config["type"])    
    working_directory = cluster_config["working_directory"]
    
    pcs = PilotComputeService(execution_engine=execution_engine, working_directory=working_directory)
    for pilot_name, pilot_compute_description in cluster_config["config"].items():
        pilot_compute_description["name"] = pilot_name
        pcs.create_pilot(pilot_compute_description=pilot_compute_description)  
              
    return pcs


class MultiPilotQuantumExecutor(Executor):

    def __init__(self, cluster_config):
        super().__init__()
        self.cluster_config = cluster_config
        self.type = self.cluster_config["type"]        
        self.pcs = initialize_client(self.cluster_config)
        self.pilots = self.pcs.get_pilots()

    def close(self):
        self.pcs.cancel()
        
    def submit_tasks(self, compute_func, *args, **kwargs):
        if self.type == "dask":
            return self.submit_tasks_dask(compute_func, *args, **kwargs)
        elif self.type == "ray":
            return self.submit_tasks_ray(compute_func, *args, **kwargs)
        
    def submit_task(self, compute_func, *args, **kwargs):
        return self.pcs.submit_task(compute_func, *args, **kwargs)
    
    def submit_tasks_dask(self, compute_func, *args, **kwargs):
        futures = []
        circuits_observables = args[0]
        circuit_bag = db.from_sequence(circuits_observables)
        args = args[1:]
        for circuit in circuit_bag:
            future = self.submit_task(compute_func, circuit, *args, **kwargs)
            futures.append(future)
            
        return futures
            
        # return circuit_bag.map(lambda x: self.submit_task(compute_func, x, *args))


    def submit_tasks_ray(self, compute_func, *args, **kwargs):
        input_tasks = args[0]  # The first argument is the collection of tasks
        args = args[1:]  # Remove the first argument
        
        return [self.submit_task(compute_func, task, *args, **kwargs) for task in input_tasks]


    def wait(self, futures):
        self.pcs.wait_tasks(futures)
        
    def get_results(self, futures):
        return self.pcs.get_results(futures)

