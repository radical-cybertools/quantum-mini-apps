import numpy as np
import os

# Load the numpy array from the file
file_path = '/global/homes/f/fkiwit/dev/quantum-mini-apps/src/mini_apps/qml_data_compression/utils/cifar10.npy'
data = np.load(file_path)

print("Shape of the data:", data.shape)

# Create the directory if it doesn't exist
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

# Save each sample separately
for index, sample in enumerate(data):
    if index % 1000 == 0:
        print(f"Saving sample {index}")
    output_path = os.path.join(output_dir, f'{index}.npy')
    np.save(output_path, sample)
