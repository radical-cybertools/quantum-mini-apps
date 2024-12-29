# Qiskit GPU from Source on Perlmutter

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