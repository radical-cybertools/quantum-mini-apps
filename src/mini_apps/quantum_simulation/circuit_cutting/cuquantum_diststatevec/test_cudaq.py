import cudaq
# from mpi4py import MPI

# print(cudaq.get_targets())
cudaq.set_target("nvidia", option="mgpu")
cudaq.mpi.initialize()


@cudaq.kernel
def ghz(numQubits:int):
    qubits = cudaq.qvector(numQubits)
    h(qubits.front())
    for i, qubit in enumerate(qubits.front(numQubits - 1)):
        x.ctrl(qubit, qubits[i + 1])

#counts = cudaq.sample(ghz, 30, execution_mode=cudaq.ExecutionMode.MPI)
counts = cudaq.sample(ghz, 30)

num_ranks = cudaq.mpi.size()
rank = cudaq.mpi.rank()
print(f"rank: {rank}, num_ranks: {num_ranks}")
if rank == 0:
    for bits, count in counts.items():
        print('Observed {} {} times.'.format(bits, count))

cudaq.mpi.finalize()