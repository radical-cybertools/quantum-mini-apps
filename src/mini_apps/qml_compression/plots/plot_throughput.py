import glob
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

benchmark_runs = [
    ["data_2_64_20240923_123316", "result_compression_20240923_123057.csv"],
    ["data_2_128_20240923_125205", "result_compression_20240923_125129.csv"],
    ["data_2_256_20240923_131124", "result_compression_20240923_131032.csv"],
    ["data_2_512_20240923_132830", "result_compression_20240923_132758.csv"]
]

# Load all files that end with *times.npy in dir

# times_data = [np.load(file, allow_pickle=True) for file in times_files]

def get_times_data(dir):
    dir = f"/pscratch/sd/f/fkiwit/work_2/{dir}"
    print(dir)
    times = []
    times_files = glob.glob(f"{dir}/*times.npy")
    for file in times_files:
        times_data = str(np.load(file, allow_pickle=True))
        times_data = ast.literal_eval(times_data)

        times.append(times_data)

    times = pd.DataFrame(times)

    times.to_csv(f"{dir}.csv", index=False)

    times['time_loading'] = times['time_sweeping_start'] - times['time_start_loading']
    times['time_sweep'] = times['time_bfgs_start'] - times['time_sweeping_start']
    times['time_bfgs'] = times['time_done'] - times['time_bfgs_start']
    
    times_mean = times[['time_loading', 'time_sweep', 'time_bfgs']].mean()
    return times_mean

labels = [128, 256, 512, 1024]
labels_position = np.arange(len(labels))
fig, axs = plt.subplots(2, 1)

"""
Preliminary plot: This plot shows the mean execution time of the compression workflow for two phases (sweeping and BFGS) on two CPU nodes. Each node ran a single Ray worker, and the benchmarks varied the number of logical CPUs per worker node (64, 128, 256, and 512). The x-axis shows the total number of logical CPUs across both nodes (2x the per-node count), and the y-axis represents execution time in seconds (upper plot) and the ratio of mean execution time to logical CPUs (lower plot). The results indicate that the highest throughput was achieved with 256 logical CPUs per node.
"""

ratios = []
for i, run in enumerate(benchmark_runs):
    times_mean = get_times_data(run[0])
    axs[0].bar(labels_position[i], times_mean['time_sweep'], bottom=0, color="tab:blue")
    axs[0].bar(labels_position[i], times_mean['time_bfgs'], bottom=times_mean['time_sweep'], color="tab:orange")
    ratios.append((times_mean['time_bfgs'] + times_mean['time_sweep']) / labels[i])

axs[0].set_ylabel("<Time> [s]")
axs[0].set_xticklabels([])
axs[0].tick_params(axis='x', which='both', bottom=False, top=False)
axs[0].legend(["<Time> Sweep", "<Time> BFGS"], loc="upper left")
axs[0].set_xlim(-0.5, 3.5)
axs[1].set_xlim(-0.5, 3.5)

# Empty plot
axs[1].plot(labels_position, ratios, marker='o')
axs[1].set_xticks(labels_position)
axs[1].set_xticklabels(labels)
axs[1].set_xlabel("Logical CPUs")
axs[1].set_ylabel("<Time> / logical CPUs [s]")

axs[0].set_title("2 Nodes with 2x AMD EPYC 7763 (512 threads total)")
plt.tight_layout()
plt.savefig("times.pdf")
