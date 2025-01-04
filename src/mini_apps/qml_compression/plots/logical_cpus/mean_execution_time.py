import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

runs = [
    "data_2_64_20240923_123316.csv",
    "data_2_128_20240923_125205.csv",
    "data_2_256_20240923_131124.csv",
    "data_2_512_20240923_132830.csv"
]

def get_times_data(dir):
    print(dir)
    times = pd.read_csv(dir)
    print(len(times))

    times['time_loading'] = times['time_sweeping_start'] - times['time_start_loading']
    times['time_sweep'] = times['time_bfgs_start'] - times['time_sweeping_start']
    times['time_bfgs'] = times['time_done'] - times['time_bfgs_start']
    
    times_mean = times[['time_loading', 'time_sweep', 'time_bfgs']].mean()
    return times_mean, len(times)

labels = [128, 256, 512, 1024]
labels_position = np.arange(len(labels))

fig, axs = plt.subplots(2, 1)
ratios = []
throughtput = []
for i, run in enumerate(runs):
    times_mean, n_processed = get_times_data(run)
    print(f"labels i {labels[i]}")
    if labels[i] == 256:
        print(times_mean['time_bfgs'] / (times_mean['time_sweep'] + times_mean['time_bfgs']))
    axs[0].bar(labels_position[i], times_mean['time_sweep'], bottom=0, color="tab:blue")
    axs[0].bar(labels_position[i], times_mean['time_bfgs'], bottom=times_mean['time_sweep'], color="tab:orange")
    ratios.append((times_mean['time_bfgs'] + times_mean['time_sweep']) / labels[i])
    throughtput.append(n_processed / (15 * 60))


axs[0].set_ylabel("<Time> [s]")
axs[0].set_xticklabels([])
axs[0].tick_params(axis='x', which='both', bottom=False, top=False)
axs[0].legend(["<Time> Sweep", "<Time> BFGS"], loc="upper left")
axs[0].set_xlim(-0.5, 3.5)
axs[1].set_xlim(-0.5, 3.5)

# Empty plot
axs[1].plot(labels_position, throughtput, marker='o')
axs[1].set_xticks(labels_position)
axs[1].set_xticklabels(labels)
axs[1].set_xlabel("Logical CPUs")
axs[1].set_ylabel("Throughput [s^-1]")

axs[0].set_title("2 Nodes with 2x AMD EPYC 7763 (512 threads total)")
plt.savefig("times.pdf")
