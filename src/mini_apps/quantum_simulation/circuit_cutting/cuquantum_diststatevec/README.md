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

        salloc --account m4408 --nodes 1 --qos interactive  --time 04:00:00 --constraint gpu --gpus 4


### Test CuQuantum 24.08 Container

* Log into container

        shifter --image=nvcr.io/nvidia/cuquantum-appliance:24.08-x86_64 /bin/bash

        shifter --image=nvcr.io/nvidia/quantum/cuda-quantum:cu12-0.9.1 /bin/bash

        * Run test script

        export MPICH_GPU_SUPPORT_ENABLED=1
        /opt/conda/envs/cuquantum-24.08/bin/python



* Run test script from outside container

        
    * MPICH
        $ export MPICH_GPU_SUPPORT_ENABLED=1
       

        $ export LD_LIBRARY_PATH=/opt/conda/envs/cuquantum-24.08/lib:/opt/conda/envs/cuquantum-24.08/lib/python3.11/site-packages/cuquantum/lib:/opt/udiImage/modules/gpu/lib64:/opt/udiImage/modules/mpich
        $ export MPICH_GPU_SUPPORT_ENABLED=1
        
        $ srun -n 2  --mpi=pmix shifter --env LD_LIBRARY_PATH=$LD_LIBRARY_PATH --image=nvcr.io/nvidia/cuquantum-appliance:24.08-x86_64 --module=cuda-mpich /opt/conda/envs/cuquantum-24.08/bin/python test_qiskit_cuquantum.py

        $ srun -n 2  --mpi=pmix shifter --env=/opt/conda/envs/cuquantum-24.08/lib:/opt/conda/envs/cuquantum-24.08/lib/python3.11/site-packages/cuquantum/lib:/opt/udiImage/modules/gpu/lib64:/opt/udiImage/modules/mpich --image=nvcr.io/nvidia/cuquantum-appliance:24.08-x86_64 --module=cuda-mpich /opt/conda/envs/cuquantum-24.08/bin/python test_qiskit_cuquantum.py




srun -n 1 --mpi=pmix shifter --env LD_LIBRARY_PATH=/opt/conda/envs/cuquantum-24.08/lib:/opt/udiImage/modules/gpu/lib64:/opt/udiImage/modules/mpich --image=nvcr.io/nvidia/cuquantum-appliance:24.08-cuda11.8.0-devel-ubuntu22.04-x86_64 --module=cuda-mpich bash -c "export LD_LIBRARY_PATH=/opt/conda/envs/cuquantum-24.08/lib:/opt/udiImage/modules/gpu/lib64:/opt/udiImage/modules/mpich && /opt/conda/envs/cuquantum-24.08/bin/python test_qiskit_cuquantum.py"


        
   

### Test CUDA-Q 24.11 Container (works)

* Container:
        * nvcr.io/nvidia/quantum/cuda-quantum:cu11-0.9.1
    
* Examplerun
        $ export MPICH_GPU_SUPPORT_ENABLED=1
        $ srun -n 1  --mpi=pmix shifter --image=nvcr.io/nvidia/quantum/cuda-quantum:cu11-0.9.1 --module=cuda-mpich python test_cudaq.py
        
    
    
    
    
    

<!--
/usr/openmpi/lib/
--volume "/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/extras/CUPTI/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/extras/CUPTI/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/extras/Debugger/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/extras/Debugger/lib64;/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/nvvm/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/nvvm/lib64;
        /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64;
        /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/nvvm/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/nvvm/lib64"
-->

### SLURM

        #!/bin/bash

        ACCOUNT="m4408"
        QOS="interactive"
        CONSTRAINT="gpu"
        NUM_NODES=1
        GPUS_PER_TASK=1
        GPUS_PER_NODE=4

        REGISTRY="nvcr.io/nvidia"
        IMAGE_NAME="cuquantum-appliance"
        IMAGE_TAG="24.08-x86_64"

        NUM_GPUS=$((${GPUS_PER_NODE}*${NUM_NODES}))

        srun --account=${ACCOUNT} \
            --qos=${QOS} \
            --constraint=${CONSTRAINT} \
            --nodes=${NUM_NODES} \
            --gpus-per-node=${GPUS_PER_NODE} \
            --gpus-per-task=${GPUS_PER_TASK} \
            --gpu-bind=none \
            --ntasks=${NUM_GPUS} \
            --gpus=${NUM_GPUS} \
            shifter --image="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
            --module=cuda-mpich \
            python qiskit_cuquantum_ghz.py

## Old notes:

        cd [/path/to/test_qiskit_cuquantum.py]
        module load cudatoolkit/11.7
        export MPICH_GPU_SUPPORT_ENABLED=1
        srun -n 2 --mpi=pmi2 --module=gpu  shifter --env LD_LIBRARY_PATH=/opt/conda/envs/cuquantum-24.08/lib:/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH --env MPICH_GPU_SUPPORT_ENABLED=1  --image=nvcr.io/nvidia/cuquantum-appliance:24.08-x86_64 /opt/conda/envs/cuquantum-24.08/bin/python `pwd`/test_qiskit_cuquantum.py


         * OpenMPI
        $ srun -n 2 --mpi=pmi2 --module=gpu  shifter --env LD_LIBRARY_PATH=/opt/conda/envs/cuquantum-24.08/lib:/usr/local/cuda-11/lib64:/usr/openmpi/lib/:$LD_LIBRARY_PATH --image=nvcr.io/nvidia/cuquantum-appliance:24.08-x86_64  /opt/conda/envs/cuquantum-24.08/bin/python  -c "from mpi4py import MPI"
