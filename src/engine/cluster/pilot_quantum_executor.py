import os, dask, ray
from pilot.pilot_compute_service import PilotComputeService
from pilot.pilot_enums_exceptions import ExecutionEngine
from distributed import Client, wait
import dask.bag as db
from engine.cluster.base_executor import Executor
from engine.cluster.dask_executor import DaskExecutor   
import psutil
import subprocess
import functools




class PilotQuantumExecutor(Executor):

    # Default excludes for Ray runtime environment to avoid uploading large files
    _default_ray_excludes = [
        ".git/**",
        ".git/lfs/**",
        ".venv/**",
        "__pycache__/**",
        "*.npy",
        "*.csv",
        ".idea/**",
        ".vscode/**",
        "slurm_output/**",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache/**",
        "*.egg-info/**",
        "dist/**",
        "build/**",
        "*.so",
        "*.dylib",
        "*.dll",
    ]

    def __init__(self, cluster_config=None):
        super().__init__()
        self.cluster_config = cluster_config or {}
        self.type = self.cluster_config["config"]["type"]
        
        # Patch ray.init to add excludes if using Ray
        if self.type == "ray":
            self._patch_ray_init()
        
        self.pilot, self.client = self.initialize_client(self.cluster_config["config"])

    def _patch_ray_init(self):
        """Patch ray.init to automatically add excludes for large files."""
        original_ray_init = ray.init
        
        @functools.wraps(original_ray_init)
        def patched_ray_init(*args, **kwargs):
            # Get excludes from cluster_config or use defaults
            excludes = self.cluster_config.get("config", {}).get("runtime_env", {}).get("excludes", self._default_ray_excludes)
            
            # Merge excludes into runtime_env
            if kwargs.get("runtime_env") is None:
                kwargs["runtime_env"] = {}
            
            # Merge excludes, avoiding duplicates
            existing_excludes = kwargs["runtime_env"].get("excludes", [])
            if isinstance(existing_excludes, list):
                combined_excludes = list(set(existing_excludes + excludes))
            else:
                combined_excludes = excludes
            
            kwargs["runtime_env"]["excludes"] = combined_excludes
            
            return original_ray_init(*args, **kwargs)
        
        # Replace ray.init with our patched version
        ray.init = patched_ray_init

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


