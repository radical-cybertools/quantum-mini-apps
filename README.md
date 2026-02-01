# Quantum Mini-Apps

This repository contains a framework for developing and benchmarking Quantum Mini-Apps, which are small, self-contained applications designed to evaluate the performance of quantum computing systems and algorithms.

## Overview

The Quantum Mini-Apps framework provides a modular and extensible architecture for defining and executing quantum computing motifs, which are fundamental building blocks or patterns of quantum algorithms. The framework leverages Qiskit and Pennylane for quantum circuit simulation and Dask/Ray for parallel and distributed execution.

The main components of the framework are:

1. **Mini-Apps**: Mini-Apps are high-level applications that combine one or more motifs to perform a specific quantum computing task or benchmark. A motif captures recurring execution patterns. For example, the Circuit Execution mini-app executes a quantum circuit on a quantum simulator or hardware backend.

2. **Executor**: The executor component manages the execution of motifs on different computing environments, such as local machines, clusters, or cloud resources. It supports different execution backends, including Dask and Ray.

3. **QDreamer**: An intelligent optimization framework for quantum circuit cutting that automatically detects hardware resources and optimizes circuit partitioning for maximum parallel execution speedup.



Architecture
----

<img src="https://github.com/radical-cybertools/quantum-mini-apps/blob/934ddc3e3dd3f4fafe9e8a1e1558e2c3cd446e4a/docs/mini-app-arch.png" alt="Mini App Architecture diagram" width="400" style="display: block; margin: auto;">

## Getting Started

To get started with the Quantum Mini-Apps framework, follow these steps:

1. Clone the repository:
```commandline
git clone https://github.com/radical-cybertools/quantum-mini-apps.git
```

2. Install the required dependencies and framework defined in ```pyproject.toml``` in Python env:

   **Using pip:**
   ```bash
   cd quantum-mini-apps
   # Optional: create and activate virtual environment with Python 3.12
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   pip install --upgrade .
   ```

   **Using uv (faster alternative):**
   ```bash
   cd quantum-mini-apps
   # uv can create and manage the venv automatically with Python 3.12
   uv venv --python 3.12  # Creates .venv directory
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   uv pip install --upgrade .
   ```

   **Note:** For GPU support (qiskit-aer-gpu), install with the `gpu` extra:
   ```bash
   # With pip
   pip install --upgrade .[gpu]
   
   # With uv
   uv pip install --upgrade .[gpu]
   ```

3. Set PYTHONPATH for easier debugging:

```
export PYTHONPATH=$PWD/src:$PYTHONPATH # Add this statement to shell startup script (like .bashrc)
```

4. Run an Mini-App:

```commandline
python src/mini_apps/quantum_simulation/circuit_execution/ce_local.py
```

This will execute the `QuantumSimulation` Mini-App with the default configuration, which runs a circuit execution motif on a local Dask cluster.

## Configuration

Before running mini-apps on HPC systems (e.g., NERSC Perlmutter), you need to configure the following settings in the example scripts:

### Required Settings

| Setting | Description | Where to Set |
|---------|-------------|--------------|
| `project` | Your HPC project allocation ID (e.g., `m1234`) | Python scripts: replace `<YOUR_PROJECT_ID>` |
| `conda_environment` | Path to your conda environment | Python scripts: replace `<CONDA_ENV_PATH>` |

### Optional Environment Variables

The following environment variables can be set to customize data and output paths:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Directory containing training/test data | `./data` |
| `RESULTS_DIR` | Directory for storing results | `./results` |
| `WORK_DIR` | Working directory for intermediate files | `./work` |
| `QML_DATA_DIR` | Directory for QML compression data | `./data` |
| `CIFAR10_PATH` | Path to CIFAR-10 dataset file | `./cifar10.npy` |
| `SCHEDULER_FILE` | Path to Dask scheduler file | `./scheduler_file.json` |

**Example:**
```bash
export DATA_DIR=/path/to/your/data
export RESULTS_DIR=/path/to/your/results
export WORK_DIR=/scratch/your_username/work
```

### SLURM Batch Scripts

For batch submission scripts in `src/mini_apps/qml_compression/`, update the `--account` directive:
```bash
#SBATCH --account=<YOUR_PROJECT_ID>  # Replace with your allocation
```

## Extending the Mini-App framework
Contributions to the Quantum Mini-Apps framework are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request. 

## Mini Apps
Each Mini-App provides in-depth documentation. The following Mini-Apps are currently implemented:

### Quantum Simulation

[**Circuit Execution**](src/mini_apps/quantum_simulation/circuit_execution/README.md) - Execute quantum circuits on simulators or hardware backends.

[**Circuit Cutting**](src/mini_apps/quantum_simulation/circuit_cutting/README.md) - Decompose large quantum circuits into smaller subcircuits for parallel execution with configurable backends and GPU acceleration.

[**QDreamer**](src/mini_apps/quantum_simulation/circuit_cutting/qdreamer/README.md) - Resource-aware quantum circuit cutting optimization framework that:
- Automatically detects available hardware resources (GPUs, CPUs, memory)
- Analyzes quantum circuit characteristics (number of qubits, gates)
- Optimizes circuit partitioning to maximize parallel execution speedup

**Examples:**
- [Basic Analysis](src/mini_apps/quantum_simulation/circuit_cutting/qdreamer/examples/basic.py) - Standalone optimization without executor
- [Executor Integration](src/mini_apps/quantum_simulation/circuit_cutting/qdreamer/examples/executor.py) - Complete pipeline with Pilot-Quantum
- [Model Calibration](src/mini_apps/quantum_simulation/circuit_cutting/qdreamer/examples/calibration.py) - Full calibration with measurements

[**State Vector Mini-Apps**](src/mini_apps/quantum_simulation/distributed_state_vector/README.md) - Distributed state vector simulation using MPI.


### Quantum Machine Learning 

[**QML Classifier**](src/mini_apps/qml_classifier/README.md)

[**QML Compression**](src/mini_apps/qml_compression/README.md)

[**QML Training**](src/mini_apps/qml_training/README.md)


## References
- Pilot-Quantum: [https://github.com/radical-cybertools/pilot-quantum](https://github.com/radical-cybertools/pilot-quantum)
- QuGEN Framework: [https://github.com/QutacQuantum/qugen](https://github.com/QutacQuantum/qugen)
- Saurabh, N., et al. "Quantum Mini-Apps: A Framework for Developing and Benchmarking Quantum-HPC Applications" [arXiv:2412.18519](https://arxiv.org/abs/2412.18519)
- Saurabh, N., et al. "Pilot-Quantum: A Quantum-HPC Middleware for Resource, Workload and Task Management" [arXiv:2405.07333](https://arxiv.org/abs/2405.07333)
- Saurabh, N., et al. "A Conceptual Architecture for a Quantum-HPC Middleware" [arXiv:2308.06608](https://arxiv.org/abs/2308.06608)



## License

This project is licensed under the [MIT License](LICENSE).



