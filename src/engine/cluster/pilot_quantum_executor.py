import os, dask, ray
from pilot.pilot_compute_service import PilotComputeService
from pilot.pilot_enums_exceptions import ExecutionEngine
from distributed import Client, wait
import dask.bag as db
from engine.cluster.base_executor import Executor
from engine.cluster.dask_executor import DaskExecutor   





class PilotQuantumExecutor(Executor):

    def __init__(self, cluster_config=None):
        super().__init__()
        self.cluster_config = cluster_config or {}
        self.type = self.cluster_config["config"]["type"]
        self.pilot, self.client = self.initialize_client(self.cluster_config["config"])

    def initialize_client(self, cluster_config):
        pilot_compute_description = cluster_config
        execution_engine = ExecutionEngine(cluster_config["type"])
        working_directory = cluster_config["working_directory"]
    
        pcs = PilotComputeService(execution_engine=execution_engine, working_directory=working_directory)
        pilot = pcs.create_pilot(pilot_compute_description=pilot_compute_description)
        pilot.wait()
        dask_ray_client = pilot.get_client()
        return pilot, dask_ray_client

    def close(self):
        if self.type == "dask":
            self.client.close()
        elif self.type == "ray":
            ray.shutdown()
        self.pilot.cancel()
        
    # def submit_mpi_task(self, number_nodes, number_procs, python_function, *args):
    #     """Run an MPI task with the given number of processes and function.
        
    #     Args:
    #         number_nodes (int): Number of nodes to use
    #         number_procs (int): Number of processes to spawn
    #         python_function (callable): Python function to execute
    #         *args: Additional arguments to pass to the function
        
    #     Returns:
    #         tuple: (stdout, stderr) from the MPI execution
    #     """
    #     import tempfile
    #     import inspect
    #     import subprocess
        
    #     # Create a temporary file to store the function and its execution
    #     with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
    #         # Write the function definition
    #         tf.write(inspect.getsource(python_function))
            
    #         # Write the execution code
    #         tf.write('\n\nif __name__ == "__main__":\n')
    #         tf.write('    import sys\n')
    #         tf.write('    args = sys.argv[1:]\n')
    #         tf.write(f'    result = {python_function.__name__}(*args)\n')
    #         tf.write('    print(result)\n')
            
    #         script_path = tf.name
            
    #     try:
    #         # Print temp file contents for debugging
    #         print(f"Generated temporary script at {script_path}:")
    #         with open(script_path, 'r') as f:
    #             print(f.read())

    #         # Execute the script using srun
    #         cmd = ["srun", "-N", str(number_nodes), "-n", str(number_procs), "python", script_path, *[str(arg) for arg in args]]
    #         print(f"Executing command: {' '.join(cmd)}")
            
    #         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
    #         print(f"Command stdout:\n{result.stdout}")
    #         print(f"Command stderr:\n{result.stderr}")
            
    #         return result.stdout, result.stderr
            
    #     finally:
    #         print(f"Cleaning up temporary file: {script_path}")
    #         os.remove(script_path)

    def submit_tasks(self, compute_func, *args, **kwargs):
        if self.type == "dask":
            return self.submit_tasks_dask(compute_func, *args,  **kwargs)
        elif self.type == "ray":
            return self.submit_tasks_ray(compute_func, *args,  **kwargs)
        
    def submit_task(self, compute_func, *args, **kwargs):
        return self.pilot.submit_task(compute_func, *args, **kwargs)
    
    # def submit_mpi_task(self, *args, **kwargs):
    #     return self.pilot.submit_mpi_task(*args, **kwargs)    

    
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

