import os, dask, ray
from pilot.pilot_compute_service import PilotComputeService
from pilot.pilot_enums_exceptions import ExecutionEngine
from distributed import Client, wait
import dask.bag as db
from engine.cluster.base_executor import Executor
from engine.cluster.dask_executor import DaskExecutor   
import psutil
import subprocess
from pilot.dreamer import QuantumTask, TaskType




class PilotQuantumExecutor(Executor):

    def __init__(self, cluster_config=None):
        super().__init__()
        self.cluster_config = cluster_config or {}
        self.type = self.cluster_config["config"]["type"]
        self.pcs, self.pilot, self.client = self.initialize_client(self.cluster_config["config"])
        self.pilots = self.pcs.get_pilots()

    def initialize_client(self, cluster_config):
        pilot_compute_description = cluster_config
        execution_engine = ExecutionEngine(cluster_config["type"])
        working_directory = cluster_config["working_directory"]
    
        pcs = PilotComputeService(execution_engine=execution_engine, working_directory=working_directory)
        pilots = []
        for pilot_compute_description in cluster_config["pilots"]:
            pilot = pcs.create_pilot(pilot_compute_description=pilot_compute_description)
            pilots.append(pilot)

        for pilot in pilots:
            pilot.wait()

        pcs.initialize_dreamer(cluster_config["dreamer_strategy"])

        dask_ray_client = pilot.get_client()
        return pcs, pilot, dask_ray_client

    def close(self):
        if self.type == "dask":
            self.client.close()
        elif self.type == "ray":
            ray.shutdown()
        self.pilot.cancel()
        self.stop_ray()
        self.kill_processes_by_keyword("pilot.plugins.ray_v2.agent")
      

    def submit_tasks(self, compute_func, *args, **kwargs):
        if self.type == "dask":
            return self.submit_tasks_dask(compute_func, *args,  **kwargs)
        elif self.type == "ray":
            return self.submit_tasks_ray(compute_func, *args,  **kwargs)
        
    def submit_task(self, compute_func, *args, **kwargs):
        return self.pilot.submit_task(compute_func, *args, **kwargs)
    
    def submit_tasks_dask(self, compute_func, *args,  **kwargs):
        circuits_observables = args[0]
        circuit_bag = db.from_sequence(circuits_observables)
        args = args[1:]
        return circuit_bag.map(lambda x: self.pilot.submit_task(compute_func, x, *args, **kwargs))


    def submit_tasks_ray(self, compute_func, *args,  **kwargs):
        input_tasks = args[0]  # The first argument is the collection of tasks
        args = args[1:]  # Remove the first argument
        
        return [self.pilot.submit_task(compute_func, task, *args,  **kwargs) for task in input_tasks]


    def submit_tasks_quantum(self, circuit_func, num_qubits, gate_set, *args,  **kwargs):
        input_tasks = args[0]  # The first argument is the collection of tasks
        args = args[1:]  # Remove the first argument
        
        return [self.submit_quantum_task(circuit_func, num_qubits, gate_set, task, *args,  **kwargs) for task in input_tasks]


    def submit_quantum_task(self, qt):        # Create quantum task        
        return self.pcs.submit_quantum_task(qt)


    def wait(self, futures):
        self.pilot.wait_tasks(futures)
        
    def get_results(self, futures):
        return self.pilot.get_results(futures)


    
    def kill_processes_by_keyword(self, keyword):
        """
        Kills all processes whose command line contains the specified keyword.
        """
        for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            try:
                # Check if the process's command line contains the keyword
                if proc.info["cmdline"] and any(
                    keyword in arg for arg in proc.info["cmdline"]
                ):
                    pid = proc.info["pid"]
                    print(
                        f"Killing process {pid} ({proc.info['name']}) with command: {proc.info['cmdline']}"
                    )
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process may have terminated or we don't have permissions
                pass


    def stop_ray(self):
        try:
            # Execute the "ray stop" command
            result = subprocess.run(
                ["ray", "stop"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print("Ray stopped successfully.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error stopping Ray:")
            print(e.stderr)


