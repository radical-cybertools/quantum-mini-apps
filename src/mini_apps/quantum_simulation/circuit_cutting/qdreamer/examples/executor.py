"""
QDreamer Executor Integration Example

This example demonstrates the complete QDreamer workflow with Pilot-Quantum executor:
1. Setup a Pilot-Quantum executor with cluster configuration
2. Create a quantum circuit
3. Use QDreamer to find optimal cutting configuration
4. Execute circuit cutting with CircuitCuttingBuilder using optimized parameters
5. Display results

Usage:
    python -m mini_apps.quantum_simulation.circuit_cutting.qdreamer.examples.executor
"""

import os
import datetime

from qiskit.circuit.library import EfficientSU2

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.circuit_cutting.motif import CircuitCuttingBuilder
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import QDreamerCircuitCutting


def create_executor(num_nodes: int = 1, cores_per_node: int = 4, gpus_per_node: int = 0):
    """
    Create and return executor with specified resources.
    
    Args:
        num_nodes: Number of nodes in the cluster
        cores_per_node: Number of CPU cores per node
        gpus_per_node: Number of GPUs per node
        
    Returns:
        Executor instance configured with the specified resources
    """
    cluster_config = {
        "executor": "pilot",
        "config": {
            "resource": "ssh://localhost",
            "working_directory": os.path.join(os.environ.get("HOME", "/tmp"), "work"),
            "type": "ray",
            "number_of_nodes": num_nodes,
            "cores_per_node": cores_per_node,
            "gpus_per_node": gpus_per_node,
        }
    }
    return MiniAppExecutor(cluster_config).get_executor()


def get_task_resources(use_gpu: bool = False):
    """Get task resource configuration based on GPU/CPU mode."""
    if use_gpu:
        return {
            'num_cpus': 1,
            'num_gpus': 0.25,  # 4 tasks per GPU
        }, {
            "device": "GPU",
            "method": "statevector",
        }
    else:
        return {
            'num_cpus': 1,
        }, {
            "device": "CPU",
            "method": "statevector",
        }


def main():
    print("=" * 70)
    print("QDreamer Executor Example")
    print("=" * 70)
    
    # Configuration
    NUM_QUBITS = 30  # Circuit size
    NUM_SAMPLES = 10000  # Sampling for circuit cutting
    USE_GPU = False
    CORES_PER_NODE = 8
    
    # Step 1: Create executor
    print("\n1. Setting up Pilot-Quantum executor...")
    executor = create_executor(
        num_nodes=1,
        cores_per_node=CORES_PER_NODE,
        gpus_per_node=4 if USE_GPU else 0,
    )
    print(f"   Executor initialized with {CORES_PER_NODE} CPU cores")
    
    # Step 2: Create quantum circuit
    print("\n2. Creating quantum circuit...")
    circuit = EfficientSU2(NUM_QUBITS, entanglement='linear', reps=2).decompose()
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
    print(f"   Circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}")
    
    # Step 3: Use QDreamer to find optimal cutting configuration
    print("\n3. Running QDreamer optimization...")
    qdreamer = QDreamerCircuitCutting(
        executor=executor,
        circuit=circuit,
        num_samples=NUM_SAMPLES,
    )
    allocation = qdreamer.optimize()
    
    print(f"\n   --- QDreamer Recommendation ---")
    print(f"   Subcircuit size: {allocation.subcircuit_size} qubits")
    print(f"   Number of cuts:  {allocation.num_cuts}")
    print(f"   Parallel tasks:  {allocation.num_parallel_tasks}")
    print(f"   Predicted speedup: {allocation.speedup_factor:.2f}x")
    
    # Check if cutting is beneficial
    if allocation.num_cuts == 0:
        print(f"\n   QDreamer recommends NOT using circuit cutting.")
        print(f"   Running full circuit simulation instead would be faster.")
        print("\n4. Skipping circuit cutting execution (not beneficial)...")
    else:
        print(f"\n   QDreamer recommends circuit cutting with {allocation.num_cuts} cut(s).")
        
        # Step 4: Execute circuit cutting with optimized parameters
        print("\n4. Executing circuit cutting with optimized configuration...")
        
        task_resources, backend_opts = get_task_resources(USE_GPU)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            os.environ.get("HOME", "/tmp"),
            "work",
            f"qdreamer_executor_example_{timestamp}.csv"
        )
        
        # Build and run circuit cutting
        cc = (
            CircuitCuttingBuilder()
            .set_subcircuit_size(allocation.subcircuit_size)
            .set_base_qubits(NUM_QUBITS)
            .set_observables(["Z" + "I" * (NUM_QUBITS - 1)])
            .set_scale_factor(1)  # Required: scaling factor for circuit
            .set_num_samples(NUM_SAMPLES)
            .set_sub_circuit_task_resources(task_resources)
            .set_full_circuit_task_resources(task_resources)
            .set_result_file(result_file)
            .set_circuit_cutting_only(True)  # Only run cutting, skip full circuit
            .set_circuit_cutting_qiskit_options({
                "backend_options": backend_opts,
                "mpi": False
            })
            .build(executor)
        )
        
        print(f"   Running circuit cutting simulation...")
        with cc:
            results = cc.run()
        
        print(f"\n   --- Execution Results ---")
        print(f"   Results saved to: {result_file}")
        if results:
            print(f"   Expectation value: {results}")
    
    # Step 5: Cleanup
    print("\n5. Cleaning up...")
    executor.close()
    print("   Executor closed.")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
