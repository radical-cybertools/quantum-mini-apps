"""
Basic QDreamer Circuit Cutting Example

This example demonstrates QDreamer's circuit cutting optimization with two scenarios:
1. A 36-qubit circuit where cutting IS beneficial (significant speedup)
2. A 20-qubit circuit where cutting is NOT beneficial (no speedup)

The key insight is that circuit cutting provides speedup when:
- The circuit is large enough that 2^(n - n_sub) factor dominates
- There are enough parallel workers to execute subcircuit tasks efficiently

Usage:
    python -m mini_apps.quantum_simulation.circuit_cutting.qdreamer.examples.basic
"""

from qiskit.circuit.library import EfficientSU2

from mini_apps.quantum_simulation.circuit_cutting.qdreamer import (
    ResourceProfile,
    ResourceOptimizer,
)


def analyze_scenario(name: str, num_qubits: int, num_workers: int, use_gpu: bool = False):
    """Analyze a circuit cutting scenario and display results."""
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {name}")
    print(f"{'=' * 70}")
    
    # Create circuit
    print(f"\nCircuit: {num_qubits} qubits (EfficientSU2, linear entanglement)")
    circuit = EfficientSU2(num_qubits, entanglement='linear', reps=2).decompose()
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
    print(f"  Depth: {circuit.depth()}, Gates: {circuit.size()}")
    
    # Create resource profile directly (no executor needed for analysis)
    print(f"\nResources: {num_workers} {'GPU' if use_gpu else 'CPU'} workers")
    resource_profile = ResourceProfile(
        num_gpus=num_workers if use_gpu else 0,
        num_cpu_cores_physical=num_workers if not use_gpu else 4,
        available_memory_gb=48.0,  # Used to calculate max subcircuit size
        gpus_per_node=num_workers if use_gpu else 0,
        cpus_per_node=num_workers if not use_gpu else 4,
    )
    
    # Run optimization (circuit characteristics computed automatically)
    optimizer = ResourceOptimizer(
        resource_profile=resource_profile,
        circuit=circuit,
        use_gpu=use_gpu,
    )
    allocation = optimizer.find_best_configuration()
    
    # Display results
    print(f"\n--- Optimization Result ---")
    print(f"  Subcircuit size: {allocation.subcircuit_size} qubits")
    print(f"  Number of cuts:  {allocation.num_cuts}")
    print(f"  Parallel tasks:  {allocation.num_parallel_tasks}")
    print(f"  Speedup factor:  {allocation.speedup_factor:.2f}x")
    
    if allocation.num_cuts == 0:
        print(f"\n  RECOMMENDATION: Do NOT use circuit cutting")
        print(f"  Circuit cutting would be slower than running the full circuit.")
        if 'best_cutting_speedup' in allocation.metadata:
            print(f"  (Best cutting speedup would be {allocation.metadata['best_cutting_speedup']:.2f}x)")
    else:
        print(f"\n  RECOMMENDATION: Use circuit cutting with {allocation.num_cuts} cut(s)")
        print(f"  Expected {allocation.speedup_factor:.1f}x faster than full circuit simulation")
    
    return allocation


def main():
    print("=" * 70)
    print("QDreamer Circuit Cutting: Beneficial vs Non-Beneficial Scenarios")
    print("=" * 70)
    print("\nThis example shows when circuit cutting helps and when it doesn't.")
    print("The power-law model predicts speedup based on qubit reduction vs overhead.")
    
    # Scenario 1: Large circuit with multiple workers - cutting IS beneficial
    # 36 qubits cut to ~12q gives 2^24 = 16M factor, easily overcoming overhead
    result1 = analyze_scenario(
        name="36-qubit circuit with 8 workers (BENEFICIAL)",
        num_qubits=36,
        num_workers=8,
        use_gpu=False,
    )
    
    # Scenario 2: Smaller circuit with few workers - cutting is NOT beneficial
    # 20 qubits: even cutting to 10q only gives 2^10 = 1024 factor
    # With sampling overhead and limited parallelism, not worth it
    result2 = analyze_scenario(
        name="20-qubit circuit with 4 workers (NOT BENEFICIAL)",
        num_qubits=20,
        num_workers=4,
        use_gpu=False,
    )
    
    # Scenario 3: Same small circuit but with MORE workers - becomes beneficial
    result3 = analyze_scenario(
        name="20-qubit circuit with 64 workers (BENEFICIAL)",
        num_qubits=20,
        num_workers=64,
        use_gpu=False,
    )
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  36q / 8 workers:  {result1.speedup_factor:>7.1f}x speedup → {'CUT' if result1.num_cuts > 0 else 'NO CUT'}")
    print(f"  20q / 4 workers:  {result2.speedup_factor:>7.1f}x speedup → {'CUT' if result2.num_cuts > 0 else 'NO CUT'}")
    print(f"  20q / 64 workers: {result3.speedup_factor:>7.1f}x speedup → {'CUT' if result3.num_cuts > 0 else 'NO CUT'}")
    
    
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
