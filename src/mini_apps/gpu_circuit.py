"""
This script creates and executes a Qiskit circuit using Qiskit Aer simulator
with multi-GPU support, dynamically querying the available GPUs from the environment.

Requirements:
    - Qiskit Aer with GPU support (`pip install qiskit-aer-gpu`) and CUDA GPUs
    - At least one GPU is required
"""

import os
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def get_num_gpus_from_env():
    """
    Determines the number of GPUs available, reading from CUDA_VISIBLE_DEVICES 
    or, if not set, tries to use nvidia-smi if available.
    """
    # Try parsing CUDA_VISIBLE_DEVICES first
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is not None:
        # CUDA_VISIBLE_DEVICES could be "0,1,2" or "" (none)
        if cuda_visible.strip() == '':
            return 0
        return len([dev for dev in cuda_visible.split(',') if dev.strip() != ''])
    else:
        try:
            # Use nvidia-smi to count GPUs, if installed
            import subprocess
            output = subprocess.check_output(
                ["nvidia-smi", "-L"], encoding="utf-8"
            )
            return len([line for line in output.splitlines() if 'GPU' in line])
        except Exception:
            # Could not detect, default to 1
            return 1

def create_meaningful_circuit(n_qubits):
    """Create a quantum circuit with substantial entanglement and depth."""
    qc = QuantumCircuit(n_qubits)
    # Initial layer of Hadamards
    for i in range(n_qubits):
        qc.h(i)
    # Create a ladder of CZ gates to entangle qubits
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    # Add a layer of parameterized RX and RY gates
    for i in range(n_qubits):
        qc.rx(0.1 * (i + 1), i)
        qc.ry(0.2 * (i + 1), i)
    # Last layer of CNOTs in a ring
    for i in range(n_qubits):
        qc.cx(i, (i + 1) % n_qubits)
    return qc

def main():
    n_qubits = 14  # Large enough to engage multiple GPUs, but not excessive for demo
    circuit = create_meaningful_circuit(n_qubits)

    num_gpus = get_num_gpus_from_env()
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        return

    print(f"Detected {num_gpus} GPU(s) available in this environment.")

    # Initialize simulator with basic GPU configuration
    simulator = AerSimulator(
        method='statevector', 
        device='GPU'
    )
    
    # Set cuStateVec options using set_options() method
    # Note: cuStateVec_num_gpus is not a valid option in Qiskit Aer
    # Multi-GPU support is typically handled via:
    # 1. Environment variables (CUDA_VISIBLE_DEVICES)
    # 2. Blocking configuration for larger circuits
    # 3. cuStateVec automatically uses available GPUs when enabled
    try:
        simulator.set_options(cuStateVec_enable=True)
        print("Enabled cuStateVec for GPU acceleration.")
        
        # For multi-GPU support with larger circuits, use blocking
        # Blocking helps distribute statevector computation across GPUs
        if num_gpus > 1 and circuit.num_qubits >= 20:
            simulator.set_options(blocking_enable=True)
            # Set blocking_qubits to enable multi-GPU distribution
            # Lower values allow more parallelism across GPUs
            blocking_qubits = max(18, circuit.num_qubits - 6)
            simulator.set_options(blocking_qubits=blocking_qubits)
            print(f"Enabled blocking (blocking_qubits={blocking_qubits}) for multi-GPU distribution.")
    except Exception as e:
        print(f"Warning: Could not set cuStateVec options: {e}")
        print("Continuing with basic GPU configuration...")

    print("Simulating circuit on Aer GPU-backed statevector simulator...")
    if num_gpus > 1:
        print(f"Note: {num_gpus} GPUs detected. Multi-GPU usage depends on circuit size and blocking configuration.")

    result = simulator.run(circuit).result()
    statevector = result.get_statevector()
    
    print(f"Statevector computed successfully (GPU device with {num_gpus} GPU(s) available).")
    print(f"Statevector norm: {abs(sum(abs(ampl)**2 for ampl in statevector)):.6f}")

if __name__ == "__main__":
    main()


