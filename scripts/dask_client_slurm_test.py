from dask.distributed import Client

# Path to the scheduler file
scheduler_file = '/global/u1/p/prmantha/scheduler_file.json'

# Connect to the Dask cluster
client = Client(scheduler_file=scheduler_file)

# Print the status of the cluster
print(client.scheduler_info())