# This code is a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""This is benchmark program for stress testing compute resources."""
import argparse
import time
from typing import List

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random.utils import random_circuit
# from qiskit.primitives import Estimator
# from qiskit.providers import Backend
# from qiskit.providers.fake_provider import ConfigurableFakeBackend
from qiskit.quantum_info.random import random_pauli_list


# from quantum_serverless import QuantumServerless, get, distribute_task, put


# @distribute_task()
def generate_circuits(
        depth_of_recursion: int, num_qubits: int, depth_of_circuit: int, n_circuits: int
):
    """Generates random circuits."""
    circuits = [random_circuit(num_qubits, depth_of_circuit) for _ in range(n_circuits)]
    if depth_of_recursion <= 1:
        return circuits
    else:
        return circuits + generate_circuits(
            depth_of_recursion - 1, num_qubits, depth_of_circuit, n_circuits
        )


# @distribute_task()
def generate_observables(
        depth_of_recursion: int, num_qubits: int, size: int, n_observables: int
):
    """Generated random observables."""
    observables = [random_pauli_list(num_qubits, size) for _ in range(n_observables)]
    if depth_of_recursion <= 1:
        return observables
    else:
        return observables + generate_observables(depth_of_recursion - 1, num_qubits, size, n_observables)


# @distribute_task()
def generate_data(
        depth_of_recursion: int,
        num_qubits: int,
        n_entries: int,
        circuit_depth: int = 2,
        size_of_observable: int = 2,
):
    return generate_circuits(
        depth_of_recursion=depth_of_recursion,
        num_qubits=num_qubits,
        n_circuits=n_entries,
        depth_of_circuit=circuit_depth,
    ), generate_observables(
        depth_of_recursion=depth_of_recursion,
        num_qubits=num_qubits,
        size=size_of_observable,
        n_observables=n_entries,
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth_of_recursion",
        help="Depth of recursion in generating data.",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--num_qubits", help="Number of qubits used in program.", default=2, type=int
    )
    parser.add_argument("--n_entries", help="Number of circuits.", default=10, type=int)
    parser.add_argument(
        "--circuit_depth", help="Depth of circuits.", default=3, type=int
    )
    parser.add_argument(
        "--size_of_observable",
        help="Size of observables in program.",
        default=3,
        type=int,
    )
    parser.add_argument("--n_backends", help="Number of backends", default=3, type=int)
    parser.add_argument(
        "--n_graphs", help="Number of graphs to run", default=1, type=int
    )

    args = parser.parse_args()

    t0: float = time.time()
    results = run_graph(
        depth_of_recursion=args.depth_of_recursion,
        num_qubits=args.num_qubits,
        n_entries=args.n_entries,
        circuit_depth=args.circuit_depth,
        size_of_observable=args.size_of_observable,
        n_backends=args.n_backends,
    )
    runtime = time.time() - t0

    print(f"Execution time: {runtime}")
    print(f"Results: {results}")
