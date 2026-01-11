# Distributed State Vector Mini App Documentation

## Overview
The Distributed State Vector Mini App is a quantum circuit simulation tool that leverages distributed computing resources to perform quantum state vector calculations using PennyLane's lightning.gpu backend with NVIDIA cuQuantum. It supports both single-node and multi-node GPU execution through MPI.

## Key Features
- Distributed quantum state vector simulation
- Support for both CPU (lightning.qubit) and GPU (lightning.gpu) backends
- MPI-enabled parallel execution
- Configurable circuit parameters (qubits, layers, runs)
- Optional Jacobian calculation
- Performance metrics collection
- QJIT (Quantum Just-In-Time) compilation support

## Background

The distributed state vector mini-app utilizes multiple processing elements, i. e., cores, nodes, and GPU, to benchmark the computational and memory needs of quantum simulations by partitioning and distributing the state vector, i. e., the state of a quantum system. In this motif, coupling is tight and occurs between classical tasks. Updates to the state vector are done by multiplying a unitary matrix. This computation is conducted concurrently. Depending on the type of operation, only local or non-local qubits, i. e., qubits placed on different processing elements, can be affected. Operations on local qubits can be performed without data exchange, while non-local or global qubits may require significant data movement. Thus, MPI is commonly used to facilitate the communication between tasks. Examples of distributed state vectors include QULAC (CPU/GPU) and cuQuantum’s cuStateVec (GPU). Further, different programming frameworks utilize cuQuantum to provide a distributed state vector simulation, e. g., Pennylane and Qiskit. 

Distributed State Vector Mini-App implementation involves PennyLane’s ```lightning.gpu``` to assess the performance of a strongly entangling layered (SEL) circuit featuring two layers, which is frequently utilized for classification tasks. For gradient calculation, the motif use adjoint differentiation, a method designed for efficient gradient computation in quantum simulations, with lower memory and computational requirements than other methods like finite difference, which requires multiple circuit evaluations. 

## Usage

### Configuration Parameters
```python
parameters = {
    "num_runs": 3,                    # Number of simulation runs
    "n_wires": 30,                    # Number of qubits
    "n_layers": 2,                    # Number of circuit layers
    "enable_jacobian": False,         # Enable Jacobian calculation
    "diff_method": "adjoint",         # Differentiation method (adjoint, parameter-shift, None)
    "enable_qjit": False,            # Enable QJIT compilation
    "pennylane_device_config": {
        "name": "lightning.gpu",      # PennyLane device backend
        "mpi": "True",               # Enable MPI distribution
        "batch_obs": "False"         # Enable batch observations
    }
}
```

### Running the App

#### 1. Via Python API
```python
from mini_apps.quantum_simulation.distributed_state_vector.mini_app import QuantumSimulation

# Configure cluster settings
cluster_config = {
    "executor": "pilot",
    "config": {
        "number_of_nodes": 1,
        "gpus_per_node": 4,
        # ... other cluster configurations
    }
}

# Initialize and run simulation
qs = QuantumSimulation(cluster_config, parameters)
qs.run()
```

#### 2. Via Command Line
```bash
# Using MPI
mpirun -n <num_gpus> python motif.py --n-wires 30 --n-layers 2 --device lightning.gpu --mpi True

# Using SLURM
srun -N <num_nodes> -n <num_gpus> python motif.py --n-wires 30 --n-layers 2 --device lightning.gpu --mpi True
```

## Output and Metrics
The app generates performance metrics in CSV format, stored in the `results` directory.

## Circuit Details
The quantum circuit implements a Strongly Entangling Layer pattern:
- Uses PennyLane's `StronglyEntanglingLayers`
- Measures PauliZ expectation values for each qubit
- Supports differentiation through various methods (adjoint, parameter-shift)

## Hardware Requirements
- NVIDIA GPUs with cuQuantum support
- MPI-enabled environment for distributed execution
- Sufficient GPU memory for larger qubit counts


# Pennylane Lightning.GPU from Source on Perlmutter

## Installation

* Source:
    * https://pennylane.ai/blog/2023/09/distributing-quantum-simulations-using-lightning-gpu-with-NVIDIA-cuQuantum
    * https://github.com/PennyLaneAI/pennylane-lightning
    * https://discuss.pennylane.ai/t/pennylane-multi-gpu-script-fails-with-error-even-there-are-enough-gpus/3978/28


# ⚡ PennyLane Lightning GPU on Perlmutter (Cray MPICH + CUDA 12)

This repository provides instructions for setting up and running the [`pennylane-lightning[gpu]`](https://github.com/PennyLaneAI/pennylane-lightning) backend on NERSC's **Perlmutter** system. These steps are validated for `v0.41.0-rc` and are compatible with **Cray MPICH**, **CUDA 12**, and **Lightning GPU**.

---

## 📦 Requirements

- Python 3.10+ (Python 3.11 recommended)
- Pennylane Lightning (v0.41.0)
- Access to NERSC's Perlmutter system
- Cray MPICH and CUDA 12 toolchain

---

## 🔧 Setup Instructions

### 1. Load Required Modules

```
module load python/3.11
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80

cd $SCRATCH
python -m venv lgpu_env && source lgpu_env/bin/activate
```

### 2. Clone and Install Dependencies

```
git clone https://github.com/PennyLaneAI/pennylane-lightning.git
cd pennylane-lightning
git checkout latest_release ### Testing with (v0.41.0)
```

### 3. Install Lightning Qubit with CrayPE Compilers

```
python -m pip install -r requirements-dev.txt && CC=$(which cc) CXX=$(which CC) python -m pip install . --verbose 
```

### 4. Switch to Lightning GPU with MPI Support

```
PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py 
CMAKE_ARGS="-DENABLE_MPI=ON" CC=$(which mpicc) CXX=$(which mpicxx) python -m pip install . --verbose
```

### 5.  Install mpi4py with Cray MPICH

```
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

### 6. Set Library Paths for custatevec

```
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
```

### 7. 🚀 Running an MPI Job

1. Allocate Interactive GPU Job (4 GPUs)

   ```
   salloc -N 1 -c 32 --qos interactive --time 0:30:00 \
       --constraint gpu --ntasks-per-node=4 \
       --gpus-per-task=1 --gpu-bind=none --account=XYZ
   ```

2. Run Your Script with MPI

   ```
   srun -n 4 python myscript.py
   ```


<!-- * Compiler Commands:

    * use miniforge and conda-forge: <https://github.com/conda-forge/miniforge>
    
    * create conda env

    ```
    conda create --prefix=${PSCRATCH}/conda/quantum-mini-apps-qml python=3.12
    ```

    * install external MPI
    ```
    conda install -c conda-forge mpi4py mpich=4.2.*=external_* 
    ``` 


    ```
    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    pip install -r requirements.txt
    #pip install custatevec-cu12
    #PL_BACKEND="lightning_qubit" python scripts/configure_pyproject_toml.py
    ```
    
    ```
    export CUQUANTUM_SDK=${PSCRATCH}/sw/cuquantum-linux-x86_64-24.11.0.21_cuda12-archive
    export LD_LIBRARY_PATH=${CUQUANTUM_SDK}/lib:${LD_LIBRARY_PATH}
    ```

    * Compile
    ```
    PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py
    CMAKE_ARGS="-DENABLE_MPI=ON" python -m pip install -e . --config-settings editable_mode=compat -vv
    ```

    ```
    CMAKE_ARGS="-DENABLE_MPI=ON -DCMAKE_C_COMPILER=/opt/cray/pe/craype/2.7.30/bin/cc -DCMAKE_CXX_COMPILER=/opt/cray/pe/craype/2.7.30/bin/CC" python -m pip install -e . --config-settings editable_mode=compat -vv
    ``` -->

## Pennylane

* Qjit has jax as dependency...

## Mini-App Usage



* Test MPI Run of Motif

```
salloc --account xxx --nodes 2 --qos interactive  --time 04:00:00 --constraint gpu --gpus 8
```

```
srun -N 2 -n 8 python motif.py --num-runs 1 --n-wires 31 --n-layers 2 --enable-jacobian False --diff-method adjoint --device lightning.gpu --mpi True

```


## Miscellaneous

* Other maybe useful commands to try:

    * clean
    ```
    make clean
    ```


    * alternative compiler commands:

       * Adjust CMAkeList.txt to use Cray compiler
        
        ```
        set(CMAKE_C_COMPILER "/opt/cray/pe/craype/2.7.30/bin/cc")
        set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/2.7.30/bin/CC")
        ```
       * compile
        
        ```
        PL_BACKEND="lightning_gpu" python scripts/configure_pyproject_toml.py
        CMAKE_ARGS="-DENABLE_MPI=ON" python -m pip install -e . --config-settings editable_mode=compat -vv
        ```

    * howto parse a compile 

    ```
    cmake -DCMAKE_C_COMPILER=/path/to/clang -DCMAKE_CXX_COMPILER=/path/to/clang++ <source-directory>
    ```
