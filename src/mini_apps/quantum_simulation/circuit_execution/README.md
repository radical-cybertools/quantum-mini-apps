# Circuit Execution Mini-App

## Overview
The Circuit Execution Mini-App is a benchmarking tool designed to evaluate the performance of quantum circuit execution across different quantum computing backends. It supports both simulator-based execution and real quantum hardware through various providers including IonQ and IBM Quantum.

## Key Features
- Support for multiple quantum backends:
  - Qiskit Aer Simulator
  - IonQ Simulator and QPU
  - IBM Quantum Runtime
- Configurable circuit parameters:
  - Number of qubits
  - Circuit depth
  - Observable size
  - Number of circuit entries
- Distributed execution capabilities using Ray/Dask
- Comprehensive performance metrics collection
- Support for both CPU and GPU-accelerated simulation

## Background

The Circuit Execution Mini-App emphasizes leveraging loosely and medium-coupled task parallelism to optimize quantum circuit execution by utilizing multiple processing elements (PEs), both classical and quantum. For instance, circuit execution is essential in tasks like sampling and estimating expectation values. Another use case involves parameterized circuits, where the same circuit is executed with varying parameters. Distributing such tasks across different PEs can significantly improve estimation accuracy. Similarly, parallelism can be employed by partitioning different terms of a Hamiltonian across multiple PEs.

Various abstraction layers and tools support this parallelism. For example, Qiskit Aer offers multiprocessing and Dask executors at the backend device level, while Qiskit Serverless provides middleware-level support.

The circuit execution mini-app utilizes the Qiskit library to generate random quantum circuits. These circuits are executed on different Aer simulator backends, including configurations with and without GPU support. To manage tasks across multiple nodes, the mini-app leverages a distributed Dask cluster environment orchestrated by the mini-app framework.


## Usage

### Basic Configuration
```python
ce_parameters = {
    "qubits": 10,
    "num_entries": 1024,
    "circuit_depth": 1,
    "size_of_observable": 1,
    "qiskit_backend_options": {
        "method": "statevector",
        "device": "CPU",
        "cuStateVec_enable": False,
        "shots": None
    }
}
```

### Running the Mini-App
```python
from mini_apps.quantum_simulation.circuit_execution import QuantumSimulation

# Configure cluster settings
cluster_info = {
    "executor": "pilot",
    "config": {
        "resource": "slurm://localhost",
        "working_directory": "/path/to/work",
        "type": "ray",
        "number_of_nodes": 1,
        "cores_per_node": 10,
        "gpus_per_node": 0
    }
}

# Initialize and run simulation
qs = QuantumSimulation(cluster_info)
futures = qs.submit_circuits(ce_parameters)
qs.wait(futures)
qs.close()
```

## Backend Options

### Aer Simulator
```python
backend_options = {
    "method": "statevector",
    "device": "CPU",
    "cuStateVec_enable": False
}
```

### IonQ
```python
backend_options = {
    "api_key": "YOUR_IONQ_API_KEY",
    "backend": "ionq_simulator"  # or "ionq_qpu"
}
```

## Implementation Details

The mini-app consists of two main components:

1. **Circuit Execution Builder**: Configures the execution parameters and builds the circuit execution instance



2. **Circuit Execution Motif**: Handles the actual execution of quantum circuits and metrics collection


## Performance Metrics
The mini-app collects and records the following metrics:
- Timestamp
- Number of qubits
- Number of circuit entries
- Circuit depth
- Observable size
- Total run time

Results are saved in CSV format for further analysis.

## Hardware Requirements
- For CPU simulation: Multi-core processor recommended
- For GPU acceleration: NVIDIA GPU with CUDA support
- For quantum hardware execution: Valid API credentials for the respective quantum provider

## License
This project is part of the Quantum Mini-Apps framework and is licensed under the MIT License.
