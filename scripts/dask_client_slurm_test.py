import os
from dask.distributed import Client

# Path to the scheduler file - set SCHEDULER_FILE environment variable or update path
scheduler_file = os.environ.get("SCHEDULER_FILE", "./scheduler_file.json")

# Connect to the Dask cluster
client = Client(scheduler_file=scheduler_file)

# Print the status of the cluster
print(client.scheduler_info())