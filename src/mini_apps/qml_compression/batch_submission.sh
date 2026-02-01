#!/bin/bash
#SBATCH --job-name=headnode
#SBATCH --output=slurm_output/run_%A/output_%a.out
#SBATCH --error=slurm_output/run_%A/error_%a.err
#SBATCH --nodes=1
#SBATCH --account=<YOUR_PROJECT_ID>  # Set to your NERSC/HPC project allocation
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00

for i in 2 4 8 16 32; do
    python src/mini_apps/qml_data_compression/qml_compression.py --num_nodes $i
done
