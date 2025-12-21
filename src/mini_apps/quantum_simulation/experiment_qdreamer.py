"""
Experiment to compare actual speedup vs predicted speedup for circuit cutting.

This script:
1. Creates a 32-qubit circuit using EfficientSU2
2. Performs circuit cutting with different numbers of cuts (1, 2, 4, 8, 16)
3. Measures actual execution times
4. Calculates actual speedup
5. Compares with model predictions
6. Plots the results
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Qiskit imports
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_addon_cutting import (
    cut_wires,
    expand_observables,
    generate_cutting_experiments,
    partition_problem,
    reconstruct_expectation_values,
)
from qiskit_addon_cutting.automated_cut_finding import (
    DeviceConstraints,
    OptimizationParameters,
    find_cuts,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

# Local imports
from mini_apps.quantum_simulation.cut_estimator import (
    SpeedupPredictor,
    HardwareCalibration,
    get_default_cpu_calibration,
)
from mini_apps.quantum_simulation.calibration import (
    calibrate_from_results,
    save_calibration,
    load_calibration,
)

# Default backend options
DEFAULT_BACKEND_OPTIONS = {"device": "CPU", "method": "statevector"}


def create_circuit_and_observable(num_qubits: int, depth: int = 3, seed: int = 42):
    """Create a test circuit and observable."""
    circuit = efficient_su2(num_qubits, reps=depth, entanglement="linear")
    param_value = 0.4 if num_qubits <= 10 else 0.8 / np.sqrt(num_qubits)
    circuit.assign_parameters([param_value] * len(circuit.parameters), inplace=True)
    
    # Create a simple observable (Z on first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1.0)])
    
    return circuit, observable


def run_full_circuit(circuit, observable, backend_options: Dict):
    """
    Run full circuit simulation and measure execution time.
    
    Transpiles the circuit for consistency with circuit cutting workflow,
    but only execution time is used for speedup calculation (matching model).
    """
    # Transpile circuit (for consistency with circuit cutting)
    transpile_backend = AerSimulator(**backend_options)
    pass_manager = generate_preset_pass_manager(
        optimization_level=0, backend=transpile_backend
    )
    
    transpile_start = time.time()
    transpiled_circuit = pass_manager.run(circuit)
    transpile_time = time.time() - transpile_start
    
    # Execute circuit
    backend = AerSimulator(**backend_options)
    estimator = EstimatorV2.from_backend(backend)
    
    execution_start = time.time()
    result = estimator.run([(transpiled_circuit, observable)])
    execution_time = time.time() - execution_start
    
    # evs is a scalar (0-dimensional array), convert to Python float
    expval = float(result.result()[0].data.evs)
    
    # Return execution time (transpile time excluded to match model)
    return execution_time, expval


def run_cut_circuit(
    circuit,
    observable,
    subcircuit_size: int,
    backend_options: Dict,
    num_samples: int = 1,
    num_workers: int = 1
) -> Tuple[float, int, int, Dict]:
    """
    Run circuit cutting and measure execution time.
    
    Returns:
        (total_time, num_cuts, num_subcircuits, total_subexperiments, max_subexperiments_per_subcircuit, metrics)
    """
    # Find cuts
    optimization_settings = OptimizationParameters(seed=111)
    device_constraints = DeviceConstraints(qubits_per_subcircuit=subcircuit_size)
    
    find_cuts_start = time.time()
    cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
    find_cuts_time = time.time() - find_cuts_start
    
    num_cuts = len(metadata["cuts"])
    
    # Prepare for cutting
    qc_w_ancilla = cut_wires(cut_circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)
    
    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )
    subcircuits = partitioned_problem.subcircuits
    print(f"Number of subcircuits: {len(subcircuits)}")
    subobservables = partitioned_problem.subobservables
    
    # Generate subexperiments
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=subcircuits, observables=subobservables, num_samples=num_samples
    )
    
    # Count total subexperiments and per-subcircuit counts
    # subexperiments is a dict: {label: [list of subexperiments for that subcircuit]}
    num_subcircuits = len(subcircuits)
    subexperiments_per_subcircuit = [len(expts) for expts in subexperiments.values()]
    max_subexperiments_per_subcircuit = max(subexperiments_per_subcircuit) if subexperiments_per_subcircuit else 1
    total_subexperiments = sum(subexperiments_per_subcircuit)
    
    print(f"Subexperiments per subcircuit: {subexperiments_per_subcircuit}")
    print(f"Max subexperiments per subcircuit: {max_subexperiments_per_subcircuit}")
    
    # Transpile subexperiments
    transpile_backend = AerSimulator(**backend_options)
    pass_manager = generate_preset_pass_manager(
        optimization_level=0, backend=transpile_backend
    )
    
    transpile_start = time.time()
    isa_subexperiments = {}
    for label, partition_subexpts in subexperiments.items():
        isa_subexperiments[label] = pass_manager.run(partition_subexpts, num_processes=1)
    transpile_time = time.time() - transpile_start
    
    # Execute subcircuits in parallel based on num_workers
    # Each subcircuit's subexperiments run sequentially within that subcircuit
    backend = AerSimulator(**backend_options)
    execution_start = time.time()
    results = {}
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from qiskit_ibm_runtime import Batch, SamplerV2
    
    def execute_subcircuit(label, subsystem_subexpts):
        """Execute all subexperiments for a single subcircuit."""
        with Batch(backend=backend) as batch:
            sampler = SamplerV2(mode=batch)
            job = sampler.run(subsystem_subexpts, shots=2**12)
            return label, job.result()
    
    # Process subcircuits in parallel batches based on num_workers
    subcircuit_items = list(isa_subexperiments.items())
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all subcircuits for parallel execution
        future_to_label = {
            executor.submit(execute_subcircuit, label, subsystem_subexpts): label
            for label, subsystem_subexpts in subcircuit_items
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_label):
            try:
                label, result = future.result()
                results[label] = result
            except Exception as e:
                label = future_to_label[future]
                print(f"Error executing subcircuit {label}: {e}")
                raise
        
    execution_time = time.time() - execution_start
    
    # Reconstruct expectation values
    reconstruction_start = time.time()
    reconstructed_expvals = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
    reconstruction_time = time.time() - reconstruction_start
    
    total_time = find_cuts_time + transpile_time + execution_time + reconstruction_time
    
    metrics = {
        'find_cuts_time': find_cuts_time,
        'transpile_time': transpile_time,
        'execution_time': execution_time,
        'reconstruction_time': reconstruction_time,
        'total_time': total_time,
    }
    
    return total_time, num_cuts, len(subcircuits), total_subexperiments, max_subexperiments_per_subcircuit, metrics


def run_experiment(
    num_qubits: int = 25,
    subcircuit_sizes: List[int] = None,
    calibration: Optional[HardwareCalibration] = None,
    num_workers: int = 1,
    backend_options: Optional[Dict] = None
):
    """
    Run the complete experiment comparing actual vs predicted speedup.
    
    This follows the qiskit addon cutting example: specify subcircuit sizes
    and let the optimizer find the cuts.
    
    Args:
        num_qubits: Number of qubits in the circuit
        subcircuit_sizes: List of subcircuit sizes to test. If None, uses default range.
        calibration: Hardware calibration parameters for the model. Defaults to CPU calibration.
        num_workers: Number of parallel workers (for model prediction)
        backend_options: Backend options for circuit execution. Defaults to CPU statevector.
    """
    backend_options = backend_options or DEFAULT_BACKEND_OPTIONS
    calibration = calibration or get_default_cpu_calibration()
    
    # Default subcircuit sizes: try a range from small to larger
    if subcircuit_sizes is None:
        # Try subcircuit sizes from 8 up to roughly half the circuit size
        subcircuit_sizes = list(range(8, min(num_qubits // 2 + 1, 20), 2))
    
    print(f"Creating {num_qubits}-qubit circuit...")
    circuit, observable = create_circuit_and_observable(num_qubits, depth=3)
    
    # Run full circuit to get baseline time
    print("Running full circuit simulation...")
    full_circuit_time, full_expval = run_full_circuit(circuit, observable, backend_options)
    print(f"Full circuit execution time: {full_circuit_time:.4f} seconds")
    print(f"Full circuit expectation value: {full_expval:.6f}")
    
    # Initialize speedup predictor
    predictor = SpeedupPredictor(calibration=calibration)
    
    # Results storage
    results = {
        'num_cuts': [],
        'actual_speedup': [],
        'predicted_speedup': [],
        'full_circuit_time': full_circuit_time,
        'cut_times': [],
        'execution_times': [],  # For calibration
        'reconstruction_times': [],  # For calibration
        'num_subexperiments': [],
        'subcircuit_sizes': [],
    }
    
    # Test different subcircuit sizes (following qiskit addon cutting example)
    for subcircuit_size in subcircuit_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with subcircuit size: {subcircuit_size}")
        print(f"{'='*60}")
        
        try:
            # Run circuit cutting with this subcircuit size
            cut_time, num_cuts, num_subcircuits, num_subexperiments, max_subexperiments_per_subcircuit, metrics = run_cut_circuit(
                circuit, observable, subcircuit_size, backend_options, num_workers=num_workers
            )
            
            print(f"Found {num_cuts} cuts")
            print(f"Number of subcircuits: {num_subcircuits}")
            print(f"Total subexperiments: {num_subexperiments}")
            print(f"Max subexperiments per subcircuit: {max_subexperiments_per_subcircuit}")
            print(f"Cut circuit execution time: {cut_time:.4f} seconds")
            print(f"  - Find cuts: {metrics['find_cuts_time']:.4f}s")
            print(f"  - Transpile: {metrics['transpile_time']:.4f}s")
            print(f"  - Execute: {metrics['execution_time']:.4f}s")
            print(f"  - Reconstruct: {metrics['reconstruction_time']:.4f}s")
            
            # Calculate actual speedup using same time components as model
            # Model uses: T_total = t_par + t_rec (execution_time + reconstruction_time)
            model_time = metrics['execution_time'] + metrics['reconstruction_time']
            actual_speedup = full_circuit_time / model_time if model_time > 0 else 0.0
            
            # Predict speedup using model
            # M = number of subcircuits (for parallel execution)
            # Within each subcircuit, subexperiments run sequentially
            # So we use: M = num_subcircuits, and account for sequential subexperiments
            predicted_speedup = predictor.predict_speedup(
                n=num_qubits,
                n_sub=subcircuit_size,
                M=num_subcircuits,  # Number of subcircuits (parallel units)
                W=num_workers,
                k=num_cuts,
                M_seq=max_subexperiments_per_subcircuit  # Sequential subexperiments per subcircuit
            )
            
            # Debug: Get breakdown to understand the prediction
            breakdown = predictor.predict_speedup(
                n=num_qubits,
                n_sub=subcircuit_size,
                M=num_subcircuits,
                W=num_workers,
                k=num_cuts,
                M_seq=max_subexperiments_per_subcircuit,
                return_breakdown=True
            )
            
            print(f"Actual speedup: {actual_speedup:.4f}")
            print(f"Predicted speedup: {predicted_speedup:.10f}")
            print(f"  Model breakdown: t_sub={breakdown['subcircuit_time']:.8f}, "
                  f"t_par={breakdown['parallel_time']:.8f}, "
                  f"t_rec={breakdown['reconstruction_time']:.8f}, "
                  f"T_total={breakdown['total_time']:.8f}")
            print(f"  Parameters: M={num_subexperiments}, W={num_workers}, k={num_cuts}, "
                  f"n={num_qubits}, n_sub={subcircuit_size}")
            
            results['num_cuts'].append(num_cuts)
            results['actual_speedup'].append(actual_speedup)
            results['predicted_speedup'].append(predicted_speedup)
            results['cut_times'].append(cut_time)
            results['execution_times'].append(metrics['execution_time'])
            results['reconstruction_times'].append(metrics['reconstruction_time'])
            results['num_subexperiments'].append(num_subexperiments)
            results['subcircuit_sizes'].append(subcircuit_size)
            
        except Exception as e:
            print(f"Error testing subcircuit_size={subcircuit_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results




def plot_results(results: Dict) -> None:
    """Plot actual vs predicted speedup."""
    if not results.get('num_cuts'):
        print("No results to plot!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_cuts = results['num_cuts']
    actual_speedup = results['actual_speedup']
    predicted_speedup = results['predicted_speedup']
    
    ax.plot(num_cuts, actual_speedup, 'o-', label='Actual Speedup', linewidth=2, markersize=8)
    ax.plot(num_cuts, predicted_speedup, 's--', label='Predicted Speedup', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Cuts', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Actual vs Predicted Speedup for Circuit Cutting', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on points
    for i, (nc, act, pred) in enumerate(zip(num_cuts, actual_speedup, predicted_speedup)):
        ax.annotate(f'{act:.2f}', (nc, act), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9)
        ax.annotate(f'{pred:.2f}', (nc, pred), textcoords="offset points", 
                   xytext=(0,-15), ha='center', fontsize=9, color='orange')
    
    plt.tight_layout()
    plt.savefig('speedup_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to 'speedup_comparison.png'")
    plt.show()


def main():
    """Main function to run the experiment."""
    print("="*60)
    print("QDreamer Speedup Prediction Experiment")
    print("="*60)
    print("\nFollowing qiskit addon cutting example: specify subcircuit sizes")
    print("and let the optimizer determine the number of cuts.\n")
    
    # Backend options - can be customized here
    backend_options = {"device": "CPU", "method": "statevector"}
    num_qubits = 25
    
    # Specify subcircuit sizes to test (following qiskit addon cutting pattern)
    # Smaller subcircuit sizes typically result in more cuts
    subcircuit_sizes = [8, 10, 12]

    # Try to load existing calibration, otherwise use default
    calibration_file = "/Users/pmantha/git_personal/quantum-mini-apps/cpu_calibration.json"
    try:
        calibration = load_calibration(calibration_file)
        print(f"Loaded calibration from {calibration_file}")
        print(f"  c_h={calibration.c_h:.4f}, e_h={calibration.e_h:.4f}, r_h={calibration.r_h:.6f}")
    except FileNotFoundError:
        print(f"No calibration file found ({calibration_file}). Using default calibration.")
        print("Run calibration first to get hardware-specific parameters.")
        calibration = None

    # Run experiment
    results = run_experiment(
        num_qubits=num_qubits,
        subcircuit_sizes=subcircuit_sizes,
        num_workers=1,  # Sequential execution for this experiment
        backend_options=backend_options,
        calibration=calibration
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Full circuit time: {results['full_circuit_time']:.4f} seconds")
    
    if not results.get('num_cuts'):
        print("\nNo results to plot. Experiment may have failed.")
        return
    
    print("\nResults by number of cuts:")
    print(f"{'Cuts':<8} {'Subcircuit':<12} {'Subexpts':<12} {'Cut Time (s)':<14} {'Actual S':<12} {'Predicted S':<12}")
    print("-" * 80)
    for i in range(len(results['num_cuts'])):
        print(f"{results['num_cuts'][i]:<8} "
              f"{results['subcircuit_sizes'][i]:<12} "
              f"{results['num_subexperiments'][i]:<12} "
              f"{results['cut_times'][i]:<14.4f} "
              f"{results['actual_speedup'][i]:<12.4f} "
              f"{results['predicted_speedup'][i]:<12.4f}")
    
    # Plot results
    plot_results(results)


if __name__ == "__main__":
    main()

