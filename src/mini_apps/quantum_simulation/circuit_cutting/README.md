# Quantum Circuit Cutting Mini-App Documentation

## Overview
The Quantum Circuit Cutting Mini-App is a benchmarking tool designed to evaluate and compare the performance of quantum circuit cutting techniques against full circuit simulation. It supports both distributed and local execution modes, with capabilities for GPU acceleration and MPI-based parallel processing.

## Key Features
- Circuit cutting with configurable subcircuit sizes
- Full circuit simulation with distributed state vector capabilities
- GPU acceleration support
- Flexible backend configuration for both cutting and full simulation
- Comprehensive metrics collection and reporting
- Support for both local and HPC (Perlmutter) environments

## Background

Circuit cutting is a technique to increase the size of circuits we can run on quantum hardware at the cost of an additional sampling overhead. A larger quantum circuit can be decomposed by cutting its gates and wires, resulting in smaller circuits that can be executed within the constraints of available quantum hardware. The results of these smaller circuits are combined to reconstruct the outcome of the original problem. Quantum Mini app framework uses Qiskit’s circuit quasiprobability decomposition (QPD) method. QPD allows the splitting of large quantum circuits into smaller sub-circuits that can be run on smaller quantum hardware or simulators with limited qubits. However, this comes with a cost: the number of times the sub-circuits need to be executed increases exponentially as the circuit size grows.

## Configuration

### Hardware Configuration
```python
BENCHMARK_CONFIG = {
    'num_runs': 3,
    'hardware_configs': [
        {
            'nodes': [1],
            'cores_per_node': 1,
            'gpus_per_node': [1]
        }
    ],
    'circuit_configs': [
        {
            'qubit_sizes': [34],
            'subcircuit_sizes': [17, 12],
            'num_samples': 1000
        }
    ]
}
```

### Backend Options
```python
CIRCUIT_CUTTING_SIMULATOR_BACKEND_OPTIONS = {
    "backend_options": {
        "device": "GPU",
        "method": "statevector",
        "shots": 4096,
        "blocking_enable": True,
        "batched_shots_gpu": True,
        "blocking_qubits": 23
    },
    "mpi": False
}
```

## Usage

### Basic Execution
```python
from mini_apps.quantum_simulation.circuit_cutting.mini_app import QuantumSimulation

# Create cluster configuration
cluster_config = create_cluster_info_perlmutter(nodes=1, cores=128, gpus=4)

# Create parameters
parameters = create_cc_parameters(
    circuit_size=34,
    subcircuit_size=17,
    num_samples=1000,
    num_nodes=1,
    num_cores=128,
    num_gpus=4
)

# Initialize and run simulation
qs = QuantumSimulation(cluster_config, parameters)
qs.run()
qs.close()
```


## Key Components

### CircuitCuttingBuilder
Builder class for configuring circuit cutting simulations with customizable settings:
- Subcircuit size
- Base qubits
- Observables
- Scale factor
- Backend options
- Resource allocation
- Result file paths

### CircuitCutting
Main class implementing the circuit cutting algorithm:
- Pre-processing for circuit cutting
- Distributed execution of subcircuits
- Full circuit simulation
- Metrics collection and reporting

## Metrics Collected
- Experiment start time
- Circuit and subcircuit sizes
- Number of tasks
- Transpilation times
- Execution times
- Circuit cutting specific metrics
- Full circuit simulation metrics
- Error estimation

## Output Format
Results are saved in CSV format with comprehensive metrics including:
- Timing information
- Resource usage
- Error measurements
- Configuration details
- Performance metrics

## Hardware Support
- Local execution
- HPC clusters (specifically Perlmutter)
- GPU acceleration
- MPI-based distributed computing

## Dependencies
- Qiskit and related packages
- Ray for distributed execution
- NumPy for numerical operations
- MPI for distributed state vector simulation

## Error Handling
The mini-app includes comprehensive error handling and logging:
- Configuration validation
- Runtime error capture
- Resource availability checks
- Execution state monitoring

## Best Practices
1. Start with smaller circuits for testing
2. Monitor GPU memory usage
3. Adjust subcircuit sizes based on available resources
4. Use appropriate backend options for your hardware
5. Enable logging for debugging

## Limitations
- GPU memory constraints for large circuits
- Overhead from circuit cutting for certain circuit topologies
- MPI scaling limitations for full circuit simulation

## Example Configuration for Perlmutter
```python
cluster_config = {
    "executor": "pilot",
    "config": {
        "resource": "slurm://localhost",
        "working_directory": "/path/to/work",
        "type": "ray",
        "number_of_nodes": 1,
        "cores_per_node": 128,
        "gpus_per_node": 4,
        "queue": "premium",
        "walltime": 30,
        "project": "m4408",
        "scheduler_script_commands": [
            "#SBATCH --constraint=gpu&hbm80g",
            "#SBATCH --gpus-per-task=1",
            "#SBATCH --ntasks-per-node=4",
            "#SBATCH --gpu-bind=none"
        ]
    }
}
```




# Qiskit GPU Compilation from Source on Perlmutter

* Source:
    * https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md


* Modules:
    
    ```
    module load conda python
    module load PrgEnv-gnu mpich cudatoolkit craype-accel-nvidia80
    ```

* Compiler Commands:

    * without cuquantum
    ```    
    conda install -c conda-forge mpi4py mpich=4.2.*=external_*
    ``` 
    
    ```
    python ./setup.py bdist_wheel -- -DAER_MPI=True -DAER_THRUST_BACKEND=CUDA 
    ```

     * Install wheel:
     ```
    pip install -U dist/*.whl
    ```

    * alternatively with cuquantum:

        * Install cuquantum: <https://developer.nvidia.com/cuQuantum-downloads>

        * Modify build script ```CMakeLists.txt```:
            * remove old ref to -lcutensor
            * ```CMAKELists.txt:``` remove `${CUDA_VERSION_MAJOR}` in path to cuquantum if you install cuquantum from tar.gz archive
        
        * Compile:
            ```
            python ./setup.py bdist_wheel -- \
                -DAER_MPI=True \
                -DAER_THRUST_BACKEND=CUDA \
                -DCUQUANTUM_ROOT=$CUQUANTUM_ROOT \                
                -DCUSTATEVEC_ROOT=$CUQUANTUM_ROOT \
                -DAER_ENABLE_CUQUANTUM=true \
                -DUSER_LIB_PATH=<PATH TO QUQUANTUM>cuquantum-linux-x86_64-24.11.0.21_cuda12-archive/lib
            ```
        * Install wheel:
              pip install -U dist/*.whl

# Run Examples

* Setup environment:

        export MPICH_GPU_SUPPORT_ENABLED=1
        export NUM_GPUS=4
        export CUQUANTUM_ROOT=/<PATH TO QUQUANTUM>/cuquantum-linux-x86_64-24.11.0.21_cuda12-archive/
        export LD_LIBRARY_PATH=$CUQUANTUM_ROOT/lib


* Single Node:

        srun -n 2 python test_qiskit_aergpu.py

        srun --ntasks-per-node=4 --gpus-per-task=1  python test_qiskit_aergpu.py 

* Multi Node:

        srun -N 2 --ntasks-per-node=4 --gpus-per-task=1  python test_qiskit_aergpu.py 

# Other things

* Cleaning

        pip uninstall qiskit-aer-gpu
        pip uninstall qiskit-aer