"""
Only works with MPI-compile Qiskit-AER-GPU
"""

import os
import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
import numpy as np
# from mpi4py import MPI


def test_ghz_circuit(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit

# Configure GPU simulator with clean parameter formatting
simulator = AerSimulator(
    method='statevector',
    device='GPU',    
    blocking_enable=True,
    blocking_qubits=24
)

# cuStateVec_enable=False,

# Test execution
print("Created simulator with GPU device")
circuit = test_ghz_circuit(n_qubits=32)
print(f"Created GHZ circuit with {circuit.num_qubits} qubits")
circuit.measure_all()
print("Added measurements to circuit")
circuit = transpile(circuit, simulator)
print("Transpiled circuit for GPU simulator")
job = simulator.run(circuit)
print("Submitted job to simulator")
result = job.result()
dict = result.to_dict()
meta = dict.get('metadata', {})
myrank = meta.get('mpi_rank', 0) if meta else None

if myrank == 0:
    print("Got result from simulator")
    print(result.get_counts())

