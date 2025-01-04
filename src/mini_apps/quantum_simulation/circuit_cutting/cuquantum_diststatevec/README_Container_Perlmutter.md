# Running distributed statevector simulation with cuQuantum appliance 24.08 on Perlmutter

## Overview

Relevant documentation:
Shifter documentation: https://docs.nersc.gov/development/containers/shifter/how-to-use/
cuQuantum documentation: https://docs.nvidia.com/cuda/cuquantum/latest/appliance/qiskit.html#getting-started
Relevant CuQuantum Issue: https://github.com/NVIDIA/cuQuantum/discussions/117

## Interactive Testing

### Shifter

Currently evaluating the following container versions: 
* nvcr.io/nvidia/cuquantum-appliance:24.08-x86_64 (not compatible due to shifter lack of cuda 12.2 support)
* nvcr.io/nvidia/cuquantum-appliance:24.08-cuda11.8.0-devel-ubuntu22.04-x86_64 (compatible with shifter?? to be tested)


<!--
* nvcr.io/nvidia/cuquantum-appliance:23.03
* nvcr.io/nvidia/cuquantum-appliance:23.10-devel-ubuntu20.04
-->

### Start SLURM Session

* allocate an interactive GPU node

        salloc --account x --nodes 1 --qos interactive  --time 04:00:00 --constraint gpu --gpus 4


### Test CuQuantum 24.08 Container

* Interactive Log into container

        shifter --image=nvcr.io/nvidia/cuquantum-appliance:24.08-x86_64 /bin/bash

        shifter --image=nvcr.io/nvidia/quantum/cuda-quantum:cu12-0.9.1 /bin/bash

        * Run test script

        export MPICH_GPU_SUPPORT_ENABLED=1
        /opt/conda/envs/cuquantum-24.08/bin/python



* Run test script from outside container

        
    * MPICH
        $ export MPICH_GPU_SUPPORT_ENABLED=1
       

        $ export LD_LIBRARY_PATH=/opt/conda/envs/cuquantum-24.08/lib:/opt/conda/envs/cuquantum-24.08/lib/python3.11/site-packages/cuquantum/lib:/opt/udiImage/modules/gpu/lib64:/opt/udiImage/modules/mpich
        
        $ srun -n 2  --mpi=pmix shifter --env LD_LIBRARY_PATH=$LD_LIBRARY_PATH --image=nvcr.io/nvidia/cuquantum-appliance:24.08-x86_64 --module=cuda-mpich /opt/conda/envs/cuquantum-24.08/bin/python test_qiskit_cuquantum.py

        $ srun -n 2  --mpi=pmix shifter --env=/opt/conda/envs/cuquantum-24.08/lib:/opt/conda/envs/cuquantum-24.08/lib/python3.11/site-packages/cuquantum/lib:/opt/udiImage/modules/gpu/lib64:/opt/udiImage/modules/mpich --image=nvcr.io/nvidia/cuquantum-appliance:24.08-x86_64 --module=cuda-mpich /opt/conda/envs/cuquantum-24.08/bin/python test_qiskit_cuquantum.py


        $ srun -n 2 \
        --mpi=pmix \
        shifter \
        --env LD_LIBRARY_PATH=/opt/udiImage/modules/gpu/lib64:/opt/udiImage/modules/mpich \
        --env CUQUANTUM_COMM_BACKEND=MPI \
        --env CUQUANTUM_ROOT=/opt/conda/envs/cuquantum-24.08/\
        --image=nvcr.io/nvidia/cuquantum-appliance:24.08-cuda11.8.0-devel-ubuntu22.04-x86_64 \
        --module=cuda-mpich \
        bash -c "export LD_LIBRARY_PATH=/opt/udiImage/modules/gpu/lib64:/opt/udiImage/modules/mpich:$LD_LIBRARY_PATH && \
        export PYTHONPATH = 
        /opt/conda/bin/conda run --prefix /opt/conda/envs/cuquantum-24.08/ python test_qiskit_cuquantum.py"

       
   

### Test CUDA-Q 24.11 Container (works)

* Container:
        * nvcr.io/nvidia/quantum/cuda-quantum:cu11-0.9.1
    
* Examplerun
        $ export MPICH_GPU_SUPPORT_ENABLED=1
        $ srun -n 1  --mpi=pmix shifter --image=nvcr.io/nvidia/quantum/cuda-quantum:cu11-0.9.1 --module=cuda-mpich python test_cudaq.py
        
    
    
    
    