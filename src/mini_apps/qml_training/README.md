Here's a README.md for the QML Training Mini-App:

# QML Training Mini-App

## Overview
The QML Training Mini-App is a quantum machine learning application designed for training quantum circuit-based generative models. It leverages the QuGEN framework and supports distributed execution through Ray/Pilot-Quantum for high-performance computing environments.

## Key Features
- Quantum Circuit Born Machine (QCBM) training
- Distributed execution capabilities
- Configurable quantum circuit parameters
- Performance metrics collection
- Support for different parallelism frameworks (JAX)


## Technical Requirements

### Software Dependencies
- Python 3.10+
- QuGEN framework
- JAX
- Pilot-Quantum
- Ray (for distributed execution)

### Hardware Requirements
- Multi-core CPU support
- Optional: GPU acceleration
- HPC cluster (for distributed training)

## Configuration

### Cluster Configuration
```python
cluster_info = {
    "executor": "pilot",
    "config": {
        "resource": "slurm://localhost",
        "type": "ray",
        "number_of_nodes": 1,
        "cores_per_node": 2,
        "queue": "debug",
        "walltime": 30
    }
}
```

### Training Parameters
```python
qml_parameters = {
    "build_parameters": {
        "model_type": "discrete",
        "data_set_name": "X_2D",
        "n_qubits": 8,
        "n_registers": 2,
        "circuit_depth": 2,
        "circuit_type": "copula",
        "transformation": "pit",
        "parallelism_framework": "jax"
    },
    "train_parameters": {
        "n_epochs": 3,
        "batch_size": 200,
        "hist_samples": 100000
    }
}
```

## Usage

### Basic Execution
```python
from mini_apps.qml_training.qml_training_miniapp import QMLTrainingMiniApp

# Initialize mini-app with configurations
qml_mini_app = QMLTrainingMiniApp(cluster_info, qml_parameters)

# Run training
qml_mini_app.run()

# Clean up
qml_mini_app.close()
```

## Implementation Details

The mini-app consists of two main components:

1. **QMLTrainingMiniApp**: Main class handling execution and metrics collection


2. **DiscreteQCBMModelHandler**: Handles the quantum circuit model building and training


## Performance Metrics
The mini-app collects and records the following metrics:
- Timestamp
- Scenario label
- Number of qubits
- Computation time
- Training parameters
- Cluster configuration
- Kullback-Leibler divergence

Results are saved in CSV format with timestamps for further analysis.


## References
- QuGEN Framework: [https://github.com/QutacQuantum/qugen](https://github.com/QutacQuantum/qugen)
- Pilot-Quantum: [https://github.com/radical-cybertools/pilot-quantum](https://github.com/radical-cybertools/pilot-quantum)
