import os

# os.environ["LD_LIBRARY_PATH"] = "/opt/conda/envs/cuquantum-24.08/lib"
# os.system("printenv LD_LIBRARY_PATH")

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from cuquantum import contract
import numpy as np
# from mpi4py import MPI
import cusvaer


# print(f"mpi4py: rank: {MPI.COMM_WORLD.Get_rank()}, size: {MPI.COMM_WORLD.Get_size()}")

def test_ghz_circuit(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit

def test_cutensor():
    a = np.random.rand(2, 2).astype(np.complex64)
    b = np.random.rand(2, 2).astype(np.complex64)
    result = contract("ij,jk->ik", a, b)
    print(result)



# Test the cuTensor library
# test_cutensor()

# Test the GHZ circuit
# simulator = Aer.get_backend('aer_simulator_statevector')
# circuit = test_ghz_circuit(n_qubits=20)
# circuit.measure_all()
# circuit = transpile(circuit, simulator)
# job = simulator.run(circuit)
# result = job.result()

# opt/nvidia/hpc_sdk/Linux_x86_64/22.7/math_libs/11.7/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/extras/CUPTI/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/extras/Debugger/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/nvvm/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64:/opt/cray/libfabric/1.20.1/lib64

options = {
    # 'device': "GPU",
    # 'cusvaer_enable': True,
    'cusvaer_comm_plugin_type': cusvaer.CommPluginType.MPI_MPICH,  # automatically select Open MPI or MPICH
    'cusvaer_comm_plugin_soname': 'libmpi.so',  # MPI library name is libmpi.so
    'cusvaer_global_index_bits': [2, 1],  # 8 devices per node, 4 nodes
    'cusvaer_p2p_device_bits': 2,         # 8 GPUs in one node
    'precision': 'double'         # use complex128
}

simulator = cusvaer.backends.StatevectorSimulator()
simulator.set_options(**options)

circuit = test_ghz_circuit(n_qubits=30)
circuit.measure_all()
job = simulator.run(circuit)
result = job.result()
if result.mpi_rank == 0:
    print(result.get_counts())
    print(f'backend: {result.backend_name}')
