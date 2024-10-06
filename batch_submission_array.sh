#!/bin/bash
#SBATCH --job-name=headnode
#SBATCH --output=slurm_output/run_%A/output_%A_%a.out
#SBATCH --error=slurm_output/run_%A/error_%A_%a.err
#SBATCH --nodes=1
#SBATCH --account=m4408
#SBATCH --qos=premium
#SBATCH --constraint=cpu
#SBATCH --time=12:00:00
#SBATCH --array=0-1

# Array of node counts (1, 2, 4, 8, 16, 32)
node_counts=(1 2)

# Get the node count for the current job array task
num_nodes=${node_counts[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the current node count
python src/mini_apps/qml_data_compression/qml_compression.py --num_nodes $num_nodes
