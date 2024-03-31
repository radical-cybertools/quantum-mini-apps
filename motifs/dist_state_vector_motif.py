from timeit import default_timer as timer

import pennylane as qml
from mpi4py import MPI
from pennylane import numpy as np

from motifs.base_motif import Motif


def run_script(**parameters):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    pennylane_device_config = parameters["pennylane_device_config"]
    num_runs = parameters["num_runs"]
    n_layers = parameters["n_layers"]
    n_wires = parameters["n_wires"]
    diff_method = parameters["diff_method"]

    # Instantiate CPU (lightning.qubit) or GPU (lightning.gpu) device
    # mpi=True to switch on distributed simulation
    # batch_obs=True to reduce the device memory demand for adjoint backpropagation
    dev = qml.device(**pennylane_device_config)

    # Create QNode of device and circuit
    @qml.qnode(dev, diff_method=diff_method)
    def circuit_adj(weights):
        qml.StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
        return qml.math.hstack([qml.expval(qml.PauliZ(i)) for i in range(n_wires)])

    # Set trainable parameters for calculating circuit Jacobian at the rank=0 process
    if rank == 0:
        params = np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
    else:
        params = None

    # Broadcast the trainable parameters across MPI processes from rank=0 process
    params = comm.bcast(params, root=0)

    # Run, calculate the quantum circuit Jacobian and average the timing results
    timing = []
    for t in range(num_runs):
        start = timer()
        jac = qml.jacobian(circuit_adj)(params)
        end = timer()
        timing.append(end - start)

    # MPI barrier to ensure all calculations are done
    comm.Barrier()

    if rank == 0:
        print("num_gpus: ", size, " wires: ", n_wires, " layers ", n_layers, " time: ", qml.numpy.mean(timing))


class DistStateVector(Motif):
    def __init__(self, executor, **parameters):
        super().__init__(executor, parameters["n_wires"])
        self.parameters = parameters

    def run(self):
        self.executor.run_sync_task(run_script, **self.parameters)
