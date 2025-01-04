# Quantum Mini-Apps

This repository contains a framework for developing and benchmarking Quantum Mini-Apps, which are small, self-contained applications designed to evaluate the performance of quantum computing systems and algorithms.

## Overview

The Quantum Mini-Apps framework provides a modular and extensible architecture for defining and executing quantum computing motifs, which are fundamental building blocks or patterns of quantum algorithms. The framework leverages the power of Qiskit for quantum circuit simulation and Dask for parallel and distributed execution.

The main components of the framework are:

1. **Mini-Apps**: Mini-Apps are high-level applications that combine one or more motifs to perform a specific quantum computing task or benchmark. A motif captures reoccuring executing patterns. For example, the Circuit Execution mini-app executes a quantum circuit on a quantum simulator or hardware backend.

2. **Executor**: The executor component manages the execution of motifs on different computing environments, such as local machines, clusters, or cloud resources. It supports different execution backends, including Dask and Ray.



Architecture
----

<img src="https://github.com/radical-cybertools/quantum-mini-apps/blob/6d5f2d2bc08b25ffa9d5ea471c552d1ed8dc3595/docs/mini-app-arch.png" alt="Mini App Architecture diagram" width="400" style="display: block; margin: auto;">

## Getting Started

To get started with the Quantum Mini-Apps framework, follow these steps:

1. Clone the repository:
```commandline
git clone https://github.com/radical-cybertools/quantum-mini-apps.git
```

2. Install the required dependencies and framework defined in ```pyproject.toml``` in Conda/Python env:
```
cd quantum-mini-apps
pip install --update .
```

3. Set PYTHONPATH for easier debugging:

```
export PYTHONPATH=$PWD/src:$PYTHONPATH # Add this statement to shell startup script (like .bashrc)
```

3. Run the provided example Mini-App:

```commandline
python src/mini_apps/quantum_simulation/circuit_execution/ce_local.py
```

This will execute the `QuantumSimulation` Mini-App with the default configuration, which runs a circuit execution motif on a local Dask cluster.

To run on Perlmutter, follow [Using Dask on Perlmutter](https://gitlab.com/NERSC/nersc-notebooks/-/tree/main/perlmutter/dask#using-dask-on-perlmutter) to provision dask cluster, and run ```python mini-apps/quantum-simulation/ce_perlmutter.py``` against the running Perlmutter dask cluster.


## Extending the Mini-App framework
Contributions to the Quantum Mini-Apps framework are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request. 

## Mini Apps
The following Mini-Apps are currently implemented:

### Quantum Simulation

[**Circuit Execution**](src/mini_apps/quantum_simulation/circuit_execution/README.md)

[**Circuit Cutting**](src/mini_apps/quantum_simulation/circuit_cutting/README.md)

[**State Vector Mini-Apps**](src/mini_apps/quantum_simulation/distributed_state_vector/README.md)


### Quantum Machine Learning 

[**QML Classifier**](src/mini_apps/qml_classifier/README.md)

[**QML Data Compression**](src/mini_apps/qml_data_compression/README.md)

[**QML Training**](src/mini_apps/qml_training/README.md)


## References
- Pilot-Quantum: [https://github.com/radical-cybertools/pilot-quantum](https://github.com/radical-cybertools/pilot-quantum)
- QuGEN Framework: [https://github.com/QutacQuantum/qugen](https://github.com/QutacQuantum/qugen)
- Saurabh, N., et al. "Quantum Mini-Apps: A Framework for Developing and Benchmarking Quantum-HPC Applications" [arXiv:2412.18519](https://arxiv.org/abs/2412.18519)
- Saurabh, N., et al. "Pilot-Quantum: A Quantum-HPC Middleware for Resource, Workload and Task Management" [arXiv:2405.07333](https://arxiv.org/abs/2405.07333)
- Saurabh, N., et al. "A Conceptual Architecture for a Quantum-HPC Middleware" [arXiv:2308.06608](https://arxiv.org/abs/2308.06608)



## License

This project is licensed under the [MIT License](LICENSE).



