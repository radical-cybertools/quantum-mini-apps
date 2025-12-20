"""
Test suite to compare CutEstimator predictions vs actual circuit execution.

Test circuits are based on:
1. EfficientSU2 ansatz (used in actual codebase)
2. RZZGate examples (from qiskit-addon-cutting documentation)
3. Various gate types mentioned in documentation (CX, RXX, RYY, etc.)

Reference: https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html#circuit-cutting-as-a-quasiprobability-decomposition-qpd
"""
import sys
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_addon_cutting import (
    cut_wires,
    expand_observables,
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)
from qiskit_ibm_runtime import Batch, SamplerV2
from cut_estimator import CutEstimator
from mock_aer_simulator import MockAerSimulator


def create_circuit_1_small_shallow(num_qubits=10):
    """Small, shallow circuit with local entanglement."""
    qc = QuantumCircuit(num_qubits)
    # Create a simple pattern: alternating layers of H and CNOT
    for i in range(0, num_qubits-1, 2):
        qc.cx(i, i+1)
    for i in range(num_qubits):
        qc.h(i)
    for i in range(1, num_qubits-1, 2):
        qc.cx(i, i+1)
    return qc


def create_circuit_2_medium_entangled(num_qubits=20):
    """Medium-sized circuit with moderate entanglement."""
    qc = QuantumCircuit(num_qubits)
    # Create a more complex entanglement pattern
    for i in range(num_qubits):
        qc.h(i)
    # Linear chain of CNOTs
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    # Add some cross connections
    for i in range(0, num_qubits-2, 2):
        qc.cx(i, i+2)
    # Another layer of Hadamards
    for i in range(0, num_qubits, 2):
        qc.h(i)
    return qc


def create_circuit_3_large_deep(num_qubits=34):
    """Large, deep circuit with extensive entanglement."""
    qc = QuantumCircuit(num_qubits)
    # Multiple layers of gates
    for layer in range(3):
        # Hadamard layer
        for i in range(num_qubits):
            qc.h(i)
        # Entanglement layer - create a ladder pattern
        for i in range(0, num_qubits-1, 2):
            qc.cx(i, i+1)
        # Cross connections
        for i in range(1, num_qubits-2, 2):
            qc.cx(i, i+1)
        # Long-range connections
        if layer < 2:
            for i in range(0, num_qubits-3, 3):
                qc.cx(i, i+3)
    return qc


def create_circuit_4_sparse(num_qubits=24):
    """Sparse circuit with minimal gates."""
    qc = QuantumCircuit(num_qubits)
    # Only a few gates scattered across the circuit
    for i in range(0, num_qubits, 4):
        qc.h(i)
    for i in range(0, num_qubits-4, 4):
        qc.cx(i, i+4)
    return qc


def create_circuit_5_highly_entangled(num_qubits=16):
    """Small but highly entangled circuit."""
    qc = QuantumCircuit(num_qubits)
    # Create a fully connected graph pattern
    for i in range(num_qubits):
        qc.h(i)
    # Connect each qubit to its neighbors and some distant ones
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    for i in range(0, num_qubits-2, 2):
        qc.cx(i, i+2)
    for i in range(0, num_qubits-4, 4):
        qc.cx(i, i+4)
    # Add more layers
    for i in range(num_qubits):
        qc.h(i)
        qc.rz(0.5, i)
    return qc


# ============================================================================
# Documentation-based test circuits
# ============================================================================

def create_circuit_efficientsu2_linear(num_qubits=30, reps=2):
    """
    EfficientSU2 circuit with linear entanglement (matches actual codebase usage).
    
    This is the same circuit type used in circuit_cutting_dreamer/motif.py.
    Reference: https://qiskit.org/documentation/stubs/qiskit.circuit.library.EfficientSU2.html
    """
    qc = EfficientSU2(num_qubits, entanglement="linear", reps=reps).decompose()
    # Assign fixed parameters (matching motif.py)
    qc.assign_parameters([0.4] * len(qc.parameters), inplace=True)
    return qc


def create_circuit_efficientsu2_full(num_qubits=20, reps=2):
    """
    EfficientSU2 circuit with full entanglement pattern.
    """
    qc = EfficientSU2(num_qubits, entanglement="full", reps=reps).decompose()
    qc.assign_parameters([0.4] * len(qc.parameters), inplace=True)
    return qc


def create_circuit_rzz_based(num_qubits=12):
    """
    Circuit with RZZGate cuts (documentation example).
    
    The documentation shows RZZGate as a key example with 6 subexperiments per cut.
    Reference: https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html#an-example-cutting-a-rzzgate
    """
    qc = QuantumCircuit(num_qubits)
    # Add initial Hadamard layer
    for i in range(num_qubits):
        qc.h(i)
    # Add RZZ gates (these are commonly cut in practice)
    # Create a linear chain with RZZ gates
    for i in range(num_qubits - 1):
        theta = 0.5 * (i + 1)  # Varying angles
        qc.rzz(theta, i, i+1)
    # Add another layer of single-qubit rotations
    for i in range(num_qubits):
        qc.ry(0.3, i)
    return qc


def create_circuit_cx_dense(num_qubits=16):
    """
    Circuit with dense CX gates (CNOT gates have overhead of 9 per cut).
    
    According to documentation, CX gates have sampling overhead of 3^2 = 9.
    """
    qc = QuantumCircuit(num_qubits)
    # Dense CX pattern
    for i in range(num_qubits):
        qc.h(i)
    # Create a ladder pattern with CX gates
    for layer in range(3):
        for i in range(0, num_qubits - 1, 2):
            qc.cx(i, i+1)
        for i in range(1, num_qubits - 1, 2):
            qc.cx(i, i+1)
    return qc


def create_circuit_rxx_ryy_mixed(num_qubits=14):
    """
    Circuit with RXX and RYY gates (parameterized gates with variable overhead).
    
    According to documentation:
    - RXX/RYY/RZZ: overhead = [1 + 2|sin(θ)|]^2
    - These gates are commonly used in variational circuits
    """
    qc = QuantumCircuit(num_qubits)
    # Initial state preparation
    for i in range(num_qubits):
        qc.h(i)
    # Add RXX gates
    for i in range(0, num_qubits - 1, 2):
        theta = 0.4 * (i + 1)
        qc.rxx(theta, i, i+1)
    # Add RYY gates
    for i in range(1, num_qubits - 1, 2):
        theta = 0.3 * (i + 1)
        qc.ryy(theta, i, i+1)
    # Add single-qubit rotations
    for i in range(num_qubits):
        qc.rz(0.2, i)
    return qc


def create_circuit_wire_cut_candidate(num_qubits=18):
    """
    Circuit designed to encourage wire cuts (time-like cuts).
    
    Wire cuts have overhead of 4^n (LO setting) or 4^n (LOCC setting, not yet supported).
    This circuit has long-range connections that may benefit from wire cutting.
    """
    qc = QuantumCircuit(num_qubits)
    # Initial layer
    for i in range(num_qubits):
        qc.h(i)
    # Long-range connections that may require wire cuts
    for i in range(0, num_qubits, 3):
        if i + 3 < num_qubits:
            qc.cx(i, i+3)
    # Another layer with different pattern
    for i in range(1, num_qubits, 3):
        if i + 3 < num_qubits:
            qc.cx(i, i+3)
    # Add some local gates
    for i in range(num_qubits):
        qc.ry(0.5, i)
    return qc


def create_circuit_deep_depth(num_qubits=50, depth=100):
    """
    Create a circuit with deep depth (many layers of gates).
    
    This circuit has significant depth to test how the model handles
    deep circuits where depth becomes a major factor in execution time.
    
    Args:
        num_qubits: Number of qubits
        depth: Number of depth layers (default 50 for deep circuit)
    """
    qc = QuantumCircuit(num_qubits)
    
    # Initial state preparation
    for i in range(num_qubits):
        qc.h(i)
    
    # Create many layers of gates to build up depth
    for layer in range(depth):
        # Alternating patterns to create entanglement and depth
        if layer % 3 == 0:
            # Linear chain of CNOTs
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
        elif layer % 3 == 1:
            # Cross connections
            for i in range(0, num_qubits - 1, 2):
                qc.cx(i, i + 1)
        else:
            # Long-range connections
            for i in range(0, num_qubits - 2, 2):
                if i + 2 < num_qubits:
                    qc.cx(i, i + 2)
        
        # Add single-qubit rotations in each layer
        for i in range(num_qubits):
            qc.ry(0.1 * (i + 1) * (layer + 1), i)
            qc.rz(0.05 * (i + 1) * (layer + 1), i)
    
    return qc


def create_circuit_parallelism_optimized(num_qubits=49, num_regions=7):
    """
    Create a circuit optimized for circuit cutting with good parallelism.
    
    This circuit is designed to benefit from cutting into multiple subcircuits
    that can run in parallel. For 6 cuts (7 subcircuits), this creates a circuit
    with a modular structure where each region can be cut relatively independently.
    
    The circuit is large enough (49 qubits) and deep enough that cutting provides
    exponential speedup benefits that outweigh the overhead.
    
    Args:
        num_qubits: Total number of qubits (should be divisible by num_regions for best results)
        num_regions: Number of regions/subcircuits (num_cuts + 1)
    
    Design principles:
    - Large size: 49 qubits so full circuit is slow (exponential scaling)
    - Modular structure: Each region has local gates with minimal cross-region connections
    - Balanced size: Each region has roughly num_qubits/num_regions qubits (~7 qubits)
    - CNOT gates: Use CNOT gates which have overhead of 9 per cut (well-understood)
    - Sufficient depth: Multiple layers to make cutting worthwhile
    - Natural cut points: Structure encourages cuts at region boundaries
    """
    qc = QuantumCircuit(num_qubits)
    qubits_per_region = num_qubits // num_regions
    
    # Layer 1: Initialize all qubits
    for i in range(num_qubits):
        qc.h(i)
    
    # Layers 2-6: Local operations within each region (creates depth)
    # This creates natural cut points between regions
    for layer in range(5):
        for region in range(num_regions):
            start_qubit = region * qubits_per_region
            end_qubit = min(start_qubit + qubits_per_region, num_qubits)
            
            # Local CNOT gates within region (creates entanglement within region)
            for i in range(start_qubit, end_qubit - 1):
                qc.cx(i, i + 1)
            
            # Single-qubit rotations
            for i in range(start_qubit, end_qubit):
                qc.ry(0.3 * (i + 1) * (layer + 1), i)
    
    # Layer 7: Minimal cross-region connections (these will be cut)
    # Only connect adjacent regions to create natural cut points
    for region in range(num_regions - 1):
        boundary_qubit = (region + 1) * qubits_per_region - 1
        next_qubit = (region + 1) * qubits_per_region
        if boundary_qubit < num_qubits and next_qubit < num_qubits:
            qc.cx(boundary_qubit, next_qubit)
    
    # Layers 8-10: More local operations within regions (adds more depth)
    for layer in range(3):
        for region in range(num_regions):
            start_qubit = region * qubits_per_region
            end_qubit = min(start_qubit + qubits_per_region, num_qubits)
            
            # RZZ gates within region (parameterized gates)
            for i in range(start_qubit, end_qubit - 1):
                theta = 0.4 * (i - start_qubit + 1) * (layer + 1)
                qc.rzz(theta, i, i + 1)
            
            # Final single-qubit rotations
            for i in range(start_qubit, end_qubit):
                qc.rz(0.2 * (i + 1) * (layer + 1), i)
    
    return qc


def execute_circuit_actual(qc, backend, shots=20_000):
    """
    Actually execute the circuit and measure time.
    
    IMPORTANT: AerSimulator automatically uses approximation methods (MPS, stabilizer)
    for circuits even when method='statevector' is specified. This is because:
    - Full statevector requires 2^N * 16 bytes of memory
    - 50 qubits = 2^50 * 16 bytes ≈ 18 PB (impossible)
    - Even 20 qubits may use approximations for efficiency
    
    This means execution times are faster than true statevector simulation would be,
    but the results use compressed representations (MPS for low-entanglement circuits).
    For truly exact simulation, you need specialized HPC resources or circuit cutting.
    """
    start_time = time.time()
    # Create estimator using the simulator
    estimator = EstimatorV2.from_backend(backend)
    # Use SparsePauliOp instead of Pauli for better compatibility with EstimatorV2
    # This matches the format used in test_nisq_vs_cutting.py
    observable = SparsePauliOp(Pauli('Z' + 'I' * (qc.num_qubits - 1)))
    
    # Remove measurements for estimator (it doesn't need them)
    # This prevents issues with EstimatorV2 when circuits have classical bits
    qc_no_measure = qc.remove_final_measurements(inplace=False)
    
    job = estimator.run([(qc_no_measure, observable)])
    result_obj = job.result()
    
    # Extract expectation value from result
    # Result structure: PrimitiveResult with PubResult objects
    # Each PubResult has data.evs (expectation values)
    evs_data = result_obj[0].data.evs
    if isinstance(evs_data, np.ndarray) and evs_data.ndim > 0:
        result = evs_data[0] if len(evs_data) > 0 else float(evs_data)
    else:
        result = float(evs_data)
    
    end_time = time.time()
    actual_time = end_time - start_time

    return actual_time, result


def execute_circuit_cutting_actual(qc, observable, backend, num_cuts, shots=20_000, num_samples=10):
    """
    Actually perform circuit cutting and execute, measuring total time.
    
    Returns:
        tuple: (total_time, num_subexperiments, actual_overhead) or None if cutting fails
    """
    try:
        from qiskit_addon_cutting.automated_cut_finding import (
            DeviceConstraints,
            OptimizationParameters,
            find_cuts,
        )
        
        total_start = time.time()
        
        # Step 1: Find cuts using automated cut finding
        # Calculate target subcircuit size based on desired number of cuts
        num_qubits = qc.num_qubits
        target_regions = num_cuts + 1
        target_subcircuit_size = max(1, num_qubits // target_regions)
        
        partition_start = time.time()
        
        # Try to find cuts using automated cut finding
        try:
            optimization_settings = OptimizationParameters(seed=111)
            device_constraints = DeviceConstraints(
                qubits_per_subcircuit=target_subcircuit_size
            )
            
            cut_circuit, metadata = find_cuts(
                qc, optimization_settings, device_constraints
            )
            
            actual_num_cuts = len(metadata["cuts"])
            if actual_num_cuts == 0:
                return None
            
        except Exception as e:
            print(f"    Error finding cuts: {e}")
            return None
        
        # Step 2: Cut wires and expand observables
        qc_w_ancilla = cut_wires(cut_circuit)
        observables_expanded = expand_observables(
            observable.paulis, qc, qc_w_ancilla
        )
        
        # Step 3: Partition the problem
        partitioned_problem = partition_problem(
            circuit=qc_w_ancilla, 
            observables=observables_expanded
        )
        partition_time = time.time() - partition_start
        
        subcircuits = partitioned_problem.subcircuits
        subobservables = partitioned_problem.subobservables
        
        # Step 3: Generate cutting experiments
        subexp_start = time.time()
        subexperiments, coefficients = generate_cutting_experiments(
            circuits=subcircuits,
            observables=subobservables,
            num_samples=num_samples
        )
        subexp_time = time.time() - subexp_start
        
        num_subexperiments = sum(len(subexp_list) for subexp_list in subexperiments.values())
        actual_overhead = num_subexperiments / 1.0  # Base is 1 (no cutting)
        
        # Step 4: Execute all subexperiments
        exec_start = time.time()
        quantum_results = {}
        
        with Batch(backend=backend) as batch:
            sampler = SamplerV2(mode=batch)
            for label, subexp_list in subexperiments.items():
                job = sampler.run(subexp_list, shots=shots)
                result = job.result()
                quantum_results[label] = result
        
        exec_time = time.time() - exec_start
        
        # Step 5: Reconstruct expectation values (optional, for validation)
        # reconstruction_start = time.time()
        # reconstructed_expval = reconstruct_expectation_values(
        #     quantum_results,
        #     coefficients,
        #     subobservables,
        # )
        # reconstruction_time = time.time() - reconstruction_start
        
        total_time = time.time() - total_start
        
        return {
            'total_time': total_time,
            'num_subexperiments': num_subexperiments,
            'actual_overhead': actual_overhead,
            'num_cuts': actual_num_cuts,
            'num_subcircuits': len(subcircuits),
            'partition_time': partition_time,
            'subexp_gen_time': subexp_time,
            'execution_time': exec_time,
        }
        
    except Exception as e:
        print(f"    Error in circuit cutting: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comparison_test(circuit_name, qc, backend, estimator, shots=20_000):
    """Run prediction and actual execution, then compare."""
    from qiskit.quantum_info import SparsePauliOp, Pauli
    
    # Get predictions
    best_cuts, info = estimator.estimate_best_cuts(qc)
    
    # Get all predictions
    predictions = {}
    for c, stats in sorted(info.items()):
        predictions[c] = {
            'subexp': stats['num_sub_experiments'],
            'overhead': stats['sampling_overhead'],
            'time': stats['total_time_sec'],
            'cost': stats['cost']
        }
    
    # Get 0 cuts prediction (full circuit)
    pred_time_0 = predictions[0]['time'] if 0 in predictions else None
    
    # Execute actual circuit (full circuit, no cutting)
    actual_time_0 = None
    try:
        actual_time_0, result = execute_circuit_actual(qc, backend, shots=shots)
    except Exception as e:
        import traceback
        error_msg = f"  ERROR executing circuit {circuit_name}: {type(e).__name__}: {e}"
        print(error_msg, file=sys.stderr)
        # Print full traceback for debugging
        traceback.print_exc(file=sys.stderr)
    
    # # Execute circuit cutting for different cut counts and compare with predictions
    # # Focus on cut counts that are in predictions and optimize for execution time
    # observable = SparsePauliOp(Pauli('Z' + 'I' * (qc.num_qubits - 1)))
    # actual_cutting_results = {}
    
    # # Test cut counts that are in predictions (excluding 0)
    # test_cut_counts = sorted([c for c in predictions.keys() if c > 0])
    
    # # Find the cut count with predicted time closest to actual full circuit time
    # # This helps identify which cut configuration might be optimal for execution time
    # if pred_time_0 is not None and actual_time_0 is not None:
    #     best_match_cuts = None
    #     best_match_diff = float('inf')
    #     for cut_count in test_cut_counts:
    #         if cut_count in predictions:
    #             pred_time = predictions[cut_count]['time']
    #             diff = abs(pred_time - actual_time_0)
    #             if diff < best_match_diff:
    #                 best_match_diff = diff
    #                 best_match_cuts = cut_count
    
    # # Execute circuit cutting for cut counts that optimize for execution time
    # # Priority: cut counts where predicted time is close to actual full circuit time
    # cut_counts_to_test = []
    
    # # 1. Add the best match (predicted time closest to actual full circuit time)
    # if best_match_cuts is not None:
    #     cut_counts_to_test.append(best_match_cuts)
    
    # # 2. Add the estimator's recommended best_cuts if different
    # if best_cuts > 0 and best_cuts not in cut_counts_to_test:
    #     cut_counts_to_test.append(best_cuts)
    
    # # 3. Add a couple more representative cut counts for comparison
    # for c in test_cut_counts:
    #     if c not in cut_counts_to_test and len(cut_counts_to_test) < 4:
    #         cut_counts_to_test.append(c)
    
    # # Execute circuit cutting for selected cut counts
    # for cut_count in cut_counts_to_test:
    #     cutting_result = execute_circuit_cutting_actual(
    #         qc, observable, backend, cut_count, shots=shots, num_samples=estimator.num_samples
    #     )
        
    #     if cutting_result:
    #         actual_cutting_results[cut_count] = cutting_result
    
    return {
        'circuit_name': circuit_name,
        'predicted_time': pred_time_0,
        'actual_time': actual_time_0,
        'num_qubits': qc.num_qubits,
        'num_gates': len(qc.data),
        'depth': qc.depth(),
        'best_cuts': best_cuts,
        'predictions': predictions,
        'actual_cutting_results': {},  # Empty dict - circuit cutting disabled for now
    }


def main():
    """Run comparison tests for multiple circuits."""
    backend = MockAerSimulator(method='statevector')
    
    estimator = CutEstimator(
        backend=backend,
        parallelism=16,
        candidate_cuts=(0, 2, 4, 6, 8, 16),
        shots=20_000,
        num_samples=100,  # Reduced to make 6 cuts more attractive (100*7=700 subexperiments)
        w_time=1.0,
        w_fidelity=0.1,
    )
    
    
    parallelism_circuits = [
        # ("Parallelism Optimized (49q, 6 cuts)", create_circuit_parallelism_optimized(49, 7)),
    ]
    
    # Group 2: Documentation-based circuits (aligned with qiskit-addon-cutting examples)
    doc_based_circuits = [
        # ("Linear Entanglement, Shallow Depth (30q)", create_circuit_efficientsu2_linear(30, reps=2)),
        # ("Linear Entanglement, Shallow Depth (20q)", create_circuit_efficientsu2_linear(20, reps=2)),
        # ("Full Entanglement, Shallow Depth (16q)", create_circuit_efficientsu2_full(16, reps=2)),
        # ("RZZ Gates, Moderate Depth (12q)", create_circuit_rzz_based(12)),
        # ("Dense CNOT, Moderate Depth (16q)", create_circuit_cx_dense(16)),
        # ("Mixed RXX/RYY, Shallow Depth (14q)", create_circuit_rxx_ryy_mixed(14)),
        # ("Long-Range Connections, Shallow Depth (18q)", create_circuit_wire_cut_candidate(18)),
        ("Deep Depth, High Entanglement (20q, 50 layers)", create_circuit_deep_depth(20, depth=50)),
    ]

    
    
    # Combine both groups - focus on parallelism-optimized circuit
    test_circuits = parallelism_circuits + doc_based_circuits
    
    results = []
    for name, qc in test_circuits:
        # Focus on full circuit prediction vs actual - circuit cutting disabled for now
        result = run_comparison_test(name, qc, backend, estimator)
        results.append(result)
    
    # Print results in table format
    print(f"\n{'='*100}")
    print(f"{'Circuit Name':<45} {'Predicted (s)':<18} {'Actual (s)':<18} {'Ratio (A/P)':<15}")
    print(f"{'-'*100}")
    
    for r in results:
        circuit_name = r['circuit_name']
        pred_time = r['predicted_time']  # This is 0 cuts (full circuit) prediction
        actual_time = r['actual_time']   # This is full circuit actual execution
        
        if pred_time is not None and actual_time is not None:
            # Ratio: actual/predicted (A/P)
            # < 1.0 means actual is faster than predicted (prediction is conservative)
            # > 1.0 means actual is slower than predicted (prediction underestimates)
            ratio = actual_time / pred_time if pred_time > 0 else float('inf')
            print(f"{circuit_name:<45} {pred_time:<18.6f} {actual_time:<18.6f} {ratio:<15.3f}x")
        elif pred_time is not None:
            print(f"{circuit_name:<45} {pred_time:<18.6f} {'ERROR':<18} {'N/A':<15}")
        elif actual_time is not None:
            print(f"{circuit_name:<45} {'N/A':<18} {actual_time:<18.6f} {'N/A':<15}")
        else:
            print(f"{circuit_name:<45} {'N/A':<18} {'ERROR':<18} {'N/A':<15}")
    
    print(f"{'='*100}")
    
    # Print circuit cutting comparison (only if results are available)
    has_cutting_results = any(r.get('actual_cutting_results') for r in results)
    
    if has_cutting_results:
        print(f"\n{'='*100}")
        print(f"CIRCUIT CUTTING EXECUTION COMPARISON (Optimized for Execution Time)")
        print(f"{'='*100}")
        print(f"{'Circuit Name':<45} {'Cuts':<8} {'Predicted (s)':<18} {'Actual (s)':<18} {'Ratio (A/P)':<15}")
        print(f"{'-'*100}")
        
        for r in results:
            circuit_name = r['circuit_name']
            predictions = r.get('predictions', {})
            actual_cutting = r.get('actual_cutting_results', {})
            actual_time_0 = r.get('actual_time')
            
            if actual_cutting:
                # Find cut count where predicted time is closest to actual full circuit time
                best_match = None
                best_match_diff = float('inf')
                
                for cut_count, cutting_result in sorted(actual_cutting.items()):
                    if cut_count in predictions:
                        pred_time = predictions[cut_count]['time']
                        actual_cut_time = cutting_result['total_time']
                        ratio = actual_cut_time / pred_time if pred_time > 0 else float('inf')
                        
                        # Check how close predicted time is to actual full circuit time
                        if actual_time_0 is not None:
                            match_diff = abs(pred_time - actual_time_0)
                            if match_diff < best_match_diff:
                                best_match_diff = match_diff
                                best_match = cut_count
                        
                        print(f"{circuit_name:<45} {cut_count:<8} {pred_time:<18.6f} {actual_cut_time:<18.6f} {ratio:<15.3f}x")
                
                if best_match is not None and actual_time_0 is not None:
                    best_pred = predictions[best_match]['time']
                    print(f"\n  → Best execution time match: {best_match} cuts")
                    print(f"     Predicted: {best_pred:.6f}s, Actual full circuit: {actual_time_0:.6f}s")
                    print(f"     Difference: {best_match_diff:.6f}s ({best_match_diff/actual_time_0*100:.1f}%)")
                    if best_match in actual_cutting:
                        actual_cut_time = actual_cutting[best_match]['total_time']
                        print(f"     Actual cutting time: {actual_cut_time:.6f}s")
                        speedup = actual_time_0 / actual_cut_time if actual_cut_time > 0 else float('inf')
                        print(f"     Speedup vs full circuit: {speedup:.2f}x")
        
        print(f"{'='*100}")
        print(f"\nNote: Ratio (A/P) = Actual / Predicted")
        print(f"  - Ratio < 1.0: Actual is faster than predicted (prediction is conservative)")
        print(f"  - Ratio > 1.0: Actual is slower than predicted (prediction underestimates)")


if __name__ == "__main__":
    main()