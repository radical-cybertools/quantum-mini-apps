import os
from timeit import default_timer as timer

from mini_apps.quantum_simulation.motifs.base_motif import Motif


RUN_SCRIPT = os.path.join(os.path.dirname(__file__), "mpi_script.py")

class DistStateVector(Motif):    
    def __init__(self, executor, parameters_file_path):
        super().__init__(executor)
        self.parameters_path = parameters_file_path

    def run(self):
        tasks = []
        for i in range(3):
            task = self.executor.submit_mpi_task(RUN_SCRIPT, "4", self.parameters_path)
            tasks.append(task)
        self.executor.wait(tasks)
        
    def cancel(self):
        self.executor.cancel()
