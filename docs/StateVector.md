Distributed state vector simulation is another motif found in the HPC-for-Quantum category. It utilizes multiple processing elements, i. e., cores, nodes, and GPU, to benchmark the computational and memory needs of quantum simulations by partitioning and distributing the state vector, i. e., the state of a quantum system. In this motif, coupling is tight and occurs between classical tasks. Updates to the state vector are done by multiplying a unitary matrix. This computation is conducted concurrently. Depending on the type of operation, only local or non-local qubits, i. e., qubits placed on different processing elements, can be affected. Operations on local qubits can be performed without data exchange, while non-local or global qubits may require significant data movement. Thus, MPI is commonly used to facilitate the communication between tasks. Examples of distributed state vectors include QULAC (CPU/GPU) and cuQuantum’s cuStateVec (GPU). Further, different programming frameworks utilize cuQuantum to provide a distributed state vector simulation, e. g., Pennylane and Qiskit. Distributed State Vector Motif implementation involves PennyLane’s ```lightning.gpu``` to assess the performance of a strongly entangling layered (SEL) circuit featuring two layers, which is frequently utilized for classification tasks. For gradient calculation, the motif use adjoint differentiation, a method designed for efficient gradient computation in quantum simulations, with lower memory and computational requirements than other methods like finite difference, which requires multiple circuit evaluations. 