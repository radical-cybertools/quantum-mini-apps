import subprocess
import json
import os
import subprocess
import json
import os
import subprocess
import json
import os
import time
from dask.distributed import Client

def launch_dask_on_slurm(num_workers, cores_per_worker, memory_per_worker, scheduler_file, num_nodes):
    # remove existing scheduler file
    if os.path.exists(scheduler_file):
        os.remove(scheduler_file)    

    # Submit SLURM job to launch Dask cluster
    job_script = f"""#!/bin/bash
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_workers}
#SBATCH --job-name=dask-cluster
#SBATCH -t 0:30:00
#SBATCH -A m4408
#SBATCH --constraint=cpu
#SBATCH -q premium

    dask scheduler --scheduler-file {scheduler_file} &
    sleep 5

    # Get the list of nodes provisioned by SLURM
    nodes = $SLURM_JOB_NODELIST
    echo $nodes

    # Launch Dask workers on each node
    for node in $(scontrol show hostname $nodes); do
        echo "Starting worker on $node"
        ssh $node "dask worker --scheduler-file {scheduler_file} --nthreads {cores_per_worker} --memory-limit {memory_per_worker}GB" &
    done

    wait
    """
    
    print(job_script.encode())
    # Submit the job and get the job IDD    
    process = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=job_script.encode())
    print(stdout.decode(), stderr.decode())

    # parse the job ID from the stdout
    job_id = stdout.decode().split()[3]
    print(f"Launched job {job_id}")

    
   # check the status of slurm job to be in running state
    while True:        
        process = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE)
        stdout, _ = process.communicate()
        if job_id in str(stdout):
            # parse the status and check if the job is running
            status = str(stdout.decode()).split('\n')[1].split()[4]
            print('Job status:', status)
            if status == 'R':
                break

        # wait for one second before checking the status again
        time.sleep(1)
    
    # check the scheduler file is created
    while not os.path.exists(scheduler_file):        
        pass

    # wait until all the workers are registered with the scheduler
    while True:
        # Connect to the Dask cluster
        client = Client(scheduler_file=scheduler_file)            
        scheduler_info = client.scheduler_info()
        print(f"Number of workers registered: {len(scheduler_info['workers'])}")
        if len(scheduler_info['workers']) == num_nodes:
            client.close()        
            return scheduler_info        
        time.sleep(1)


# Usage example
num_workers = 4
cores_per_worker = 2
memory_per_worker = 4
scheduler_file = '/global/homes/p/prmantha/scheduler_file.json'

num_nodes = 2  # Update with the desired number of nodes
scheduler_info = launch_dask_on_slurm(num_workers, cores_per_worker, memory_per_worker, scheduler_file, num_nodes)
print(scheduler_info)