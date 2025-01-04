import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

data = [
    "jTrue_vTrue_b4_20241007_010756", "jTrue_vTrue_b8_20241007_010757", "jTrue_vFalse_b32_20241007_010812",
    "jFalse_vTrue_b16_20241007_010825", "jFalse_vFalse_b2_20241007_010830", "jTrue_vTrue_b32_20241007_010838",
    "jTrue_vTrue_b2_20241007_010840", "jTrue_vFalse_b16_20241007_010851", "jFalse_vTrue_b8_20241007_010912",
    "jFalse_vTrue_b4_20241007_010918", "jFalse_vFalse_b4_20241007_010918", "jTrue_vTrue_b16_20241007_010934",
    "jFalse_vTrue_b32_20241007_010941", "jTrue_vFalse_b4_20241007_010945", "jFalse_vTrue_b2_20241007_010948",
    "jTrue_vFalse_b2_20241007_010952", "jTrue_vFalse_b8_20241007_011000", "jTrue_vFalse_b64_20241007_011012",
    "jFalse_vTrue_b64_20241007_011035", "jTrue_vTrue_b64_20241008_032811", "jFalse_vFalse_b8_20241007_011147",
    "jFalse_vFalse_b16_20241007_013110", "jFalse_vFalse_b32_20241007_013210", "jFalse_vFalse_b64_20241007_013401"
]

mean_std_file = "training_loop_data.csv"
if os.path.exists(mean_std_file):
    print(f"Reading {mean_std_file}")
    df = pd.read_csv(mean_std_file)
else:
    print(f"Creating {mean_std_file}")
    data_list = []
    for d in data:
        path = f"/pscratch/sd/f/fkiwit/work_classifier/{d}"
        with open(f"{path}/times.yml", 'r') as file:
            times_data = yaml.safe_load(file)
        with open(f"{path}/config.yml", 'r') as file:
            config = yaml.safe_load(file)

        times_training_loop = np.load(f"{path}/times_training_loop.npy")
        mean_time = np.mean(times_training_loop[2:])
        std_time = np.std(times_training_loop[2:])

        data_list.append([config["batch_size"], mean_time, std_time, d])

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data_list, columns=["batch_size", "mean_time", "std_time", "config"])
    df = df.sort_values(by="batch_size")
    df["category"] = df["config"].apply(lambda x: x.split('_')[0] + "_" + x.split('_')[1])
    df.to_csv("training_loop_data.csv", index=False)

# Batch size to position mapping (for consistent x-ticks)
batch_size_to_pos = {2: 0, 4: 1, 8: 2, 16: 3, 32: 4, 64: 5}

category_label_map = {
    'jFalse_vFalse': 'Sequential',
    'jFalse_vTrue': 'Batch-Optimized (vmap)',
    'jTrue_vFalse': 'Compiled (JIT)',
    'jTrue_vTrue': 'Batch-Opt.+Compiled'
}
categories = list(category_label_map.keys())

plt.figure()
for category in categories:
    category_data = df[df["category"] == category]

    print(category)
    print(category_data["mean_time"].values)
    print(category_data["std_time"].values)
    plt.errorbar(
        [batch_size_to_pos[i] for i in category_data["batch_size"]],
        category_data["mean_time"],
        yerr=category_data["std_time"], 
        label=category_label_map[category],  # Set label from the mapping
        marker='o'
    )

plt.xlabel('Batch Size')
plt.ylabel('Batch Processing Time [s]')
plt.legend()
plt.yscale('log')
plt.xticks(list(batch_size_to_pos.values()), list(batch_size_to_pos.keys()))
plt.savefig("training_loop_time.pdf")


    (2, 2.98183821) +- (0, 0.04838781)
    (1, 1.52332445) +- (0, 0.04751922)
    (3, 5.85643724) +- (0, 0.22458307)
    (4, 12.11635395) +- (0, 0.00420862)
    (5, 24.74481913) +- (0, 0.00527187)
    (6, 46.29661712) +- (0, 0.10428157)
