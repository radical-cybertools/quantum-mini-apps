import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

mean_std_file = 'mean_std_multinode.csv'

if os.path.exists(mean_std_file):
    print(f"Reading {mean_std_file}")
    mean_std = pd.read_csv(mean_std_file, index_col='number_of_nodes')
else:
    print(f"Creating {mean_std_file}")
    data = pd.read_csv('results_multinode.csv')
    mean_std = data.groupby('number_of_nodes')['compute_time_sec'].agg(['mean', 'std'])
    mean_std.to_csv(mean_std_file)

compute_times = mean_std['mean'].values
stds = mean_std['std'].values
nodes = mean_std.index.values
labels_position = np.arange(len(nodes))

base_time = compute_times[0]
base_time_std = stds[0]

eff_file = 'eff_std_multinode.csv'
if os.path.exists(eff_file):
    print(f"Reading {eff_file}")
    efficiency_df = pd.read_csv(eff_file)
else:
    print(f"Creating {eff_file}")
    eff = []
    eff_errors = []
    for time, std, node in zip(compute_times, stds, nodes):
        efficiency = 100 * base_time / time / node
        eff.append(efficiency)
        efficiency_error = efficiency * np.sqrt((std / time) ** 2 + (base_time_std / base_time) ** 2)
        eff_errors.append(efficiency_error)

    efficiency_df = pd.DataFrame({
        'nodes': nodes,
        'efficiency': eff,
        'efficiency_error': eff_errors
    })
    efficiency_df.to_csv(eff_file)

eff = efficiency_df['efficiency'].values
eff_errors = efficiency_df['efficiency_error'].values
sweep_times = compute_times * (1 - 0.7039053394074545)
bfgs_times = compute_times * 0.7039053394074545

print(f"sweep_times: {sweep_times}")
print(f"bfgs_times: {bfgs_times}")

fig, axs = plt.subplots(2, 1)
axs[0].set_title("Each node 2x AMD EPYC 7763")
axs[0].bar(labels_position, bfgs_times, bottom=sweep_times, color="tab:blue", label='Time BFGS')
axs[0].bar(labels_position, sweep_times, color="tab:orange", label='Time Sweep')
axs[0].legend()
axs[0].set_ylabel('Time [s]')
axs[0].set_xticklabels([])
axs[0].tick_params(axis='x', which='both', bottom=False, top=False)
axs[0].set_xlim(-0.5, len(nodes) - 0.5)

# Plot the speedup as a function of the values
axs[1].axhline(y=100, color='black', linestyle='--')
axs[1].plot(labels_position, eff, color='tab:blue')
axs[1].errorbar(labels_position, eff, yerr=eff_errors, fmt='o', color='tab:blue', markersize=0)
print(f"efficiency: {eff}")
print(f"efficiency error: {eff_errors}")
axs[1].set_xlabel('Nodes')
axs[1].set_ylabel('Efficiency [%]')
axs[1].set_xticks(labels_position)
axs[1].set_xticklabels(nodes)
axs[1].set_xlim(-0.5, len(nodes) - 0.5)
axs[1].set_ylim(0, 110)

fig.savefig('multi_node.pdf')
