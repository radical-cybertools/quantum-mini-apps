import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the paths and corresponding values
# Set RESULTS_DIR environment variable to your results directory
results_dir = os.environ.get("RESULTS_DIR", "./results")

# Example data_info structure - update paths to your result files
data_info = {
    1: f'{results_dir}/nodes1/result_compression.csv',
    2: f'{results_dir}/nodes2/result_compression.csv',
    4: f'{results_dir}/nodes4/result_compression.csv',
    8: f'{results_dir}/nodes8/result_compression.csv',
    16: f'{results_dir}/nodes16/result_compression.csv',
    32: f'{results_dir}/nodes32/result_compression.csv',
}

nodes = list(data_info.keys())
labels_position = np.arange(len(nodes))

# Initialize lists to store the values and compute times
values = []
compute_times = []

# Load each dataframe and extract the compute time
for value, path in data_info.items():
    df = pd.read_csv(path)
    compute_time_sec = df["compute_time_sec"][0]
    values.append(value)
    compute_times.append(compute_time_sec)

# Plot the compute times as a function of the values
fig, axs = plt.subplots(2, 1)
axs[0].set_title("Each node 2x AMD EPYC 7763")
axs[0].bar(labels_position, compute_times, color="tab:blue")
axs[0].set_ylabel('Time [s]')
axs[0].set_xticklabels([])
axs[0].tick_params(axis='x', which='both', bottom=False, top=False)
axs[0].set_xlim(-0.5, len(nodes) - 0.5)

# Calculate the speedup relative to the first compute time
base_time = compute_times[0]
eff = [100 * base_time / time / node for time, node in zip(compute_times, nodes)]

# Plot the speedup as a function of the values
axs[1].axhline(y=100, color='black', linestyle='--')
axs[1].plot(labels_position, eff, marker='o')
axs[1].set_xlabel('Nodes')
axs[1].set_ylabel('Efficiency [%]')
axs[1].set_xticks(labels_position)
axs[1].set_xticklabels(nodes)
axs[1].set_xlim(-0.5, len(nodes) - 0.5)
axs[1].set_ylim(0, 110)

plt.tight_layout()
fig.savefig('multi_node.pdf')
