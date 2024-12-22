# Quantum Mini-Apps

This repository contains a framework for developing and benchmarking Quantum Mini-Apps, which are small, self-contained applications designed to evaluate the performance of quantum computing systems and algorithms.

## Overview

The Quantum Mini-Apps framework provides a modular and extensible architecture for defining and executing quantum computing motifs, which are fundamental building blocks or patterns of quantum algorithms. The framework leverages the power of Qiskit for quantum circuit simulation and Dask for parallel and distributed execution.

The main components of the framework are:

1. **Mini-Apps**: Mini-Apps are high-level applications that combine one or more motifs to perform a specific quantum computing task or benchmark. A motif captures reoccuring executing patterns. For example, the Circuit Execution mini-app executes a quantum circuit on a quantum simulator or hardware backend.

2. **Executor**: The executor component manages the execution of motifs on different computing environments, such as local machines, clusters, or cloud resources. It supports different execution backends, including Dask and Ray.



Architecture
----

[mini-app-arch](https://github.com/radical-cybertools/quantum-mini-apps/files/14898257/mini-app-arch.1.pdf)


## Getting Started

To get started with the Quantum Mini-Apps framework, follow these steps:

1. Clone the repository:
```commandline
git clone https://github.com/radical-cybertools/quantum-mini-apps.git
```

2. Install the required dependencies:
```
cd quantum-mini-apps
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH ### Add this statement to shell startup script (like .bashrc)
```

3. Run the provided example Mini-App:

```commandline
python src/mini_apps/quantum_simulation/circuit_execution/ce_local.py
```

This will execute the `QuantumSimulation` Mini-App with the default configuration, which runs a circuit execution motif on a local Dask cluster.

To run on Perlmutter, follow [Using Dask on Perlmutter](https://gitlab.com/NERSC/nersc-notebooks/-/tree/main/perlmutter/dask#using-dask-on-perlmutter) to provision dask cluster, and run ```python mini-apps/quantum-simulation/ce_perlmutter.py``` against the running Perlmutter dask cluster.


## Extending the Mini-App framework
Contributions to the Quantum Mini-Apps framework are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

To customize the framework or develop your own Mini-Apps/Motifs, you can extend the base Motif class which provides executor as an abstraction for executing the mini-apps. 


## Mini Apps
The following Mini-Apps are currently implemented

### Quantum Simulation
The Quantum Simulation mini-app implements following motifs

[**Circuit Execution**](docs/CircuitExecution.md)

[**Circuit Cutting**](docs/CircuitCutting.md)

[**State Vector Mini-Apps**](docs/StateVector.md)


### QML Data Compression & Training


## License

This project is licensed under the [MIT License](LICENSE).


## Resources

[Using Dask on Perlmutter](https://gitlab.com/NERSC/nersc-notebooks/-/tree/main/perlmutter/dask#using-dask-on-perlmutter)

