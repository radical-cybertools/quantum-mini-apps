import os
import sys

import dask
from pilot.pilot_compute_service import PilotComputeService

RESOURCE_URL_HPC = "slurm://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")
NUM_QUBITS = int(sys.argv[1])
NUMBER_NODES = int(sys.argv[2])
GPUS_PER_NODE = 4
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
num_gpus = NUMBER_NODES *  GPUS_PER_NODE
SCRIPT_WITH_JACOBIAN = f"{SCRIPT_DIR}/dist_mem_no_jacobian.py"
OUTPUT_FILE = f"dist_mem_jacobian_GPU_{NUMBER_NODES}"

pilot_compute_description_dask = {
    "resource": RESOURCE_URL_HPC,
    "working_directory": WORKING_DIRECTORY,
    "queue": "regular",
    "type": "dask",
    "walltime": 180,
    "project": "m4408",
    "number_of_nodes": NUMBER_NODES,
    "scheduler_script_commands": ["#SBATCH --constraint=gpu", f"#SBATCH --gpus={num_gpus}", "#SBATCH -C \"gpu&hbm80g\""]
}

def start_pilot():
    pcs = PilotComputeService()
    dp = pcs.create_pilot(pilot_compute_description=pilot_compute_description_dask)
    dp.wait()
    return dp

def get_command():
    return f"srun -n {num_gpus} python {SCRIPT_WITH_JACOBIAN} {NUM_QUBITS} > ~/{OUTPUT_FILE} 2>&1"

if __name__ == "__main__":
    dask_pilot, dask_client = None, None

    try:
        # Start Pilot
        dask_pilot = start_pilot()

        # Get Dask client details
        dask_client = dask_pilot.get_client()
        print(dask_client.scheduler_info())

        run_func = dask.delayed(os.system)
        dask_client.gather(dask_client.compute(run_func(get_command())))
    finally:
        if dask_pilot:
            dask_pilot.cancel()
