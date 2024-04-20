# Quantum Mini-Apps

This repository contains a framework for developing and benchmarking Quantum Mini-Apps, which are small, self-contained applications designed to evaluate the performance of quantum computing systems and algorithms.

## Overview

The Quantum Mini-Apps framework provides a modular and extensible architecture for defining and executing quantum computing motifs, which are fundamental building blocks or patterns of quantum algorithms. The framework leverages the power of Qiskit for quantum circuit simulation and Dask for parallel and distributed execution.

The main components of the framework are:

1. **Motifs**: Motifs are the core components of the framework, representing specific quantum computing tasks or algorithms. Each motif is implemented as a separate class and can be configured with various parameters.

2. **Executor**: The executor component manages the execution of motifs on different computing environments, such as local machines, clusters, or cloud resources. It supports different execution backends, including Dask and Ray.

3. **Mini-Apps**: Mini-Apps are high-level applications that combine one or more motifs to perform a specific quantum computing task or benchmark. They provide a convenient interface for configuring and running the motifs with different parameters.

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
pip install -r requirements.txt
```

3. Run the provided example Mini-App:

```commandline
python mini-apps/quantum-simulation-ce.py
```

his will execute the `QuantumSimulation` Mini-App with the default configuration, which runs a circuit execution motif on a local Dask cluster.

## Customization

To customize the framework or develop your own Mini-Apps, you can follow these sample steps:

1. **Implement a new Motif**: Create a new class that inherits from the `Motif` base class and implement the `run` method to define the quantum computing task or algorithm.

2. **Configure the Motif**: Use the provided builder classes (e.g., `CircuitExecutionBuilder`) to configure the motif with the desired parameters.

3. **Create a new Mini-App**: Define a new class that inherits from the `MiniApp` base class and implement the `run` method to combine and execute the desired motifs.

4. **Configure the Execution Environment**: Modify the `cluster_info` dictionary in the `main.py` file to specify the desired execution environment (e.g., local, cluster, or cloud) and its configuration.

5. **Run the Mini-App**: Execute the created mini-app script file with the appropriate configuration to run the Mini-App and benchmark the performance of the quantum computing system.

## Contributing

Contributions to the Quantum Mini-Apps framework are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).


## Resources

[Using Dask on Perlmutter](https://gitlab.com/NERSC/nersc-notebooks/-/tree/main/perlmutter/dask#using-dask-on-perlmutter)
