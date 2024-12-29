# Pennylane Lightning.GPU from Source on Perlmutter

* Source:
    * https://pennylane.ai/blog/2023/09/distributing-quantum-simulations-using-lightning-gpu-with-NVIDIA-cuQuantum
    * https://github.com/PennyLaneAI/pennylane-lightning


* Modules:
    
    ```
    module load conda python
    module load PrgEnv-gnu mpich cudatoolkit craype-accel-nvidia80
    ```

* Compiler Commands:


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

    CMAKE_ARGS="-DENABLE_MPI=ON -DCMAKE_C_COMPILER=/opt/cray/pe/craype/2.7.30/bin/cc -DCMAKE_CXX_COMPILER=/opt/cray/pe/craype/2.7.30/bin/CC" python -m pip install -e . --config-settings editable_mode=compat -vv

    ```



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
