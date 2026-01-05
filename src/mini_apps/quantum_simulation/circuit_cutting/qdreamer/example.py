"""
QDreamer Circuit Cutting Example

This example demonstrates how to use QDreamer to optimize and execute circuit cutting
with automatically optimized parameters based on your available hardware resources.

Usage:
    python -m mini_apps.quantum_simulation.circuit_cutting.qdreamer.example
"""

import os
import datetime
import logging
import csv
import subprocess
import re
import numpy as np

from qiskit.circuit.library import EfficientSU2

from engine.manager import MiniAppExecutor
from mini_apps.quantum_simulation.circuit_cutting.motif import CircuitCuttingBuilder
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import QDreamerCircuitCutting

import sys
csv.field_size_limit(sys.maxsize)



def detect_gpu_type():
    """
    Detect the GPU type using nvidia-smi.
    
    Returns:
        str: GPU type (e.g., 'B200', 'H100', 'A100') or 'Unknown' if detection fails
    """
    try:
        # Run nvidia-smi to get GPU name
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]  # Get first GPU
            
            # Extract GPU type from name
            # Common patterns: "NVIDIA H100 PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA B200", etc.
            if 'B200' in gpu_name:
                return 'B200'
            elif 'H100' in gpu_name:
                return 'H100'
            elif 'A100' in gpu_name:
                return 'A100'
            elif 'V100' in gpu_name:
                return 'V100'
            elif 'A40' in gpu_name:
                return 'A40'
            elif 'RTX' in gpu_name:
                # Extract RTX model (e.g., RTX 4090, RTX 3090)
                match = re.search(r'RTX\s*(\d+)', gpu_name, re.IGNORECASE)
                if match:
                    return f'RTX{match.group(1)}'
                return 'RTX'
            else:
                # Return the full name if no specific pattern matches
                return gpu_name
        else:
            return 'Unknown'
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logging.warning(f"Failed to detect GPU type: {e}")
        return 'Unknown'


def create_executor(num_nodes, cores_per_node, gpus_per_node):
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
            # Exclude large files from Ray runtime environment
            "runtime_env": {
                "excludes": [
                    ".git/**",
                    ".git/lfs/**",
                    ".venv/**",
                    "__pycache__/**",
                    "*.npy",
                    "*.csv",
                    ".idea/**",
                    ".vscode/**",
                    "slurm_output/**",
                    "*.pyc",
                    "*.pyo",
                    "*.pyd",
                    ".pytest_cache/**",
                    "*.egg-info/**",
                    "dist/**",
                    "build/**",
                ]
            }
        }
    }
    
    executor = MiniAppExecutor(cluster_config).get_executor()
    return executor


# Helper functions for CSV parsing
def safe_float(value, default=0.0):
    """Safely convert string to float, handling empty strings."""
    if value == '' or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    """Safely convert string to int, handling empty strings."""
    if value == '' or value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def get_task_resources_and_backend_opts(use_gpu=False, gpu_fraction=0.25):
    """
    Get task resources and backend options based on GPU/CPU mode.
    
    Args:
        use_gpu: Whether to use GPU acceleration
        gpu_fraction: Fraction of GPU per task (only used when use_gpu=True)
    
    Returns:
        Tuple of (sub_circuit_task_resources, full_circuit_task_resources,
                 circuit_cutting_backend_opts, full_circuit_backend_opts)
    """
    if use_gpu:
        sub_circuit_task_resources = {
            'num_cpus': 1,              # CPUs per task
            'num_gpus': gpu_fraction,   # Fraction of GPU per task (e.g., 0.25 = 4 tasks per GPU)
        }
        full_circuit_task_resources = {
            'num_cpus': 1,
            'num_gpus': 1.0,            # Full circuit may need full GPU
        }
        circuit_cutting_backend_opts = {
            "device": "GPU",
            "method": "statevector",
            "shots": 4096,
            "blocking_enable": True,
            "batched_shots_gpu": True,
            "blocking_qubits": 31
        }
        full_circuit_backend_opts = {
            "device": "GPU",
            "method": "statevector",
            "shots": 4096,
            "blocking_enable": True,
            "batched_shots_gpu": True,
            "blocking_qubits": 31
        }
    else:
        sub_circuit_task_resources = {
            'num_cpus': 1,              # More CPUs when not using GPU
            'num_threads': 2
        }
        full_circuit_task_resources = {
            'num_cpus': 1,
            'num_threads': 2
        }
        circuit_cutting_backend_opts = {
            "device": "CPU",
            "method": "statevector",
            "shots": 4096,
        }
        full_circuit_backend_opts = {
            "device": "CPU",
            "method": "statevector",
            "shots": 4096,
        }
    
    return (sub_circuit_task_resources, full_circuit_task_resources,
            circuit_cutting_backend_opts, full_circuit_backend_opts)


def run_experiment(num_qubits, executor, result_file, 
                   use_gpu=False, gpu_fraction=0.25, repeat=None,
                   num_nodes=1, cores_per_node=224, gpus_per_node=4,
                   num_samples=10000, circuit_type="EfficientSU2", circuit_depth=2):
    """
    Run single QDreamer experiment for given number of qubits.

    Args:
        num_qubits: Number of qubits in the circuit
        executor: MiniAppExecutor instance
        result_file: Path to CSV file (will append results)
        use_gpu: Whether to use GPU acceleration (default: False)
        gpu_fraction: Fraction of GPU to request per task (default: 0.25)
        repeat: Repeat number for this experiment (optional, for tracking multiple runs)
        num_nodes: Number of nodes in the cluster (for CSV tracking)
        cores_per_node: Number of CPU cores per node (for CSV tracking)
        gpus_per_node: Number of GPUs per node (for CSV tracking)
        num_samples: Number of samples for circuit cutting (default: 10000)
        circuit_type: Type of circuit to use (default: "EfficientSU2")
        circuit_depth: Circuit depth/reps (default: 2)

    Returns:
        dict: Experiment results summary
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {num_qubits} qubits {'(GPU Mode)' if use_gpu else '(CPU Mode)'}")
    print(f"{'='*80}")

    # Detect GPU type if using GPU
    gpu_type = detect_gpu_type() if use_gpu else None
    if use_gpu:
        print(f"Detected GPU type: {gpu_type}")

    # Create a test circuit    
    circuit = EfficientSU2(num_qubits, entanglement='linear', reps=circuit_depth).decompose()
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
    # Store EfficientSU2 metadata for circuit type detection in motif.py
    circuit._efficientsu2_reps = circuit_depth

    print(f"\nStep 1: Created {num_qubits}-qubit test circuit")
    print(f"  Circuit type: {circuit_type}, depth/reps: {circuit_depth}")
    print(f"  Gates: {len(circuit)}")
    print(f"  Depth: {circuit.depth()}")

    # Use QDreamer to optimize parameters (executor passed in)
    print("\nStep 2: Running QDreamer optimization...")
    qdreamer = QDreamerCircuitCutting(
        executor=executor,
        circuit=circuit,
        use_gpu=use_gpu
    )

    # Get optimized allocation from QDreamer
    alloc = qdreamer.optimize()
    
    # Get task resources and backend options
    (sub_circuit_task_resources, full_circuit_task_resources,
     circuit_cutting_backend_opts, full_circuit_backend_opts) = get_task_resources_and_backend_opts(
        use_gpu=use_gpu, gpu_fraction=gpu_fraction
    )

    circuit_cutting_config = {
        'subcircuit_size': alloc.subcircuit_size,
        'base_qubits': num_qubits,
        'scale_factor': 1,
        'num_samples': num_samples,
        'sub_circuit_task_resources': sub_circuit_task_resources,
        'full_circuit_task_resources': full_circuit_task_resources,
        'use_gpu': use_gpu,
        'sampling_overhead': alloc.sampling_overhead
    }

    # Store QDreamer predictions for later comparison
    predicted_speedup = alloc.speedup_factor

    print("\n" + "-" * 80)
    print("Circuit Cutting Configuration:")
    print("-" * 80)
    print(f"  Recommended Subcircuit size: {circuit_cutting_config['subcircuit_size']} qubits")
    print(f"  Expected speedup: {predicted_speedup:.3f}x")
    print(f"  Sampling overhead: {alloc.sampling_overhead:.2f}x")
    print(f"  GPU acceleration: {circuit_cutting_config['use_gpu']}")
    if use_gpu:
        print(f"    ├─ GPU fraction per subcircuit task: {gpu_fraction}")
        print(f"    └─ Max parallel tasks per GPU: {int(1/gpu_fraction)}")
    print(f"  Sub-circuit task resources: {circuit_cutting_config['sub_circuit_task_resources']}")
    print(f"  Full-circuit task resources: {circuit_cutting_config['full_circuit_task_resources']}")
    
    print("-" * 80)

    # Build and configure CircuitCutting motif
    print("\nStep 4: Building CircuitCutting motif with optimized parameters...")


    cc_builder = CircuitCuttingBuilder()
    cc = (
        cc_builder
        .set_subcircuit_size(circuit_cutting_config['subcircuit_size'])
        .set_base_qubits(circuit_cutting_config['base_qubits'])
        .set_observables(["Z" + "I" * (num_qubits - 1)])
        .set_scale_factor(circuit_cutting_config['scale_factor'])
        .set_num_samples(num_samples)
        .set_sub_circuit_task_resources(circuit_cutting_config['sub_circuit_task_resources'])
        .set_full_circuit_task_resources(circuit_cutting_config['full_circuit_task_resources'])
        .set_result_file(result_file)
        .set_circuit_cutting_qiskit_options({
            "backend_options": circuit_cutting_backend_opts,
            "mpi": False
        })
        .set_full_circuit_qiskit_options({
            "backend_options": full_circuit_backend_opts,
            "mpi": False
        })
        .set_full_circuit_only(False)
        .set_circuit_cutting_only(False)
        .set_scenario_label(f"qdreamer_optimized_{num_qubits}q_repeat{repeat}" if repeat is not None else f"qdreamer_optimized_{num_qubits}q")
        .set_qdreamer_allocation(alloc)  # Pass QDreamer allocation for unified CSV
        .set_gpu_mode(use_gpu)  # Track GPU mode in CSV
        .set_gpu_fraction(gpu_fraction)  # Track GPU fraction in CSV
        .set_gpu_type(gpu_type)  # Track GPU type in CSV
        .set_num_nodes(num_nodes)  # Track parallelization config in CSV
        .set_cores_per_node(cores_per_node)
        .set_gpus_per_node(gpus_per_node)
        .build(executor)
    )

    print("✓ CircuitCutting motif configured")

    # Execute circuit cutting
    print("\nStep 5: Running circuit cutting simulation...")
    print("=" * 80)

    with cc:
        cc.run()

    print("=" * 80)
    print(f"\n✓ Circuit cutting completed successfully!")
    print(f"\nResults saved to: {result_file}")
    print(f"  (includes QDreamer predictions and actual measurements in unified CSV)")

    # Display results summary from unified CSV

    # Read the last line from the CSV results
    with open(result_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            last_row = rows[-1]

            # Extract timing metrics with safe conversion
            circuit_cutting_time = safe_float(last_row.get('circuit_cutting_total_runtime_secs', ''))
            full_circuit_time = safe_float(last_row.get('full_circuit_total_runtime_secs', ''))
            cutting_exec = safe_float(last_row.get('circuit_cutting_exec_time_secs', ''))
            num_tasks = safe_int(last_row.get('number_of_tasks', ''))
            avg_task_time = cutting_exec / num_tasks if num_tasks > 0 else 0

            # Extract QDreamer predictions from unified CSV
            actual_speedup = safe_float(last_row.get('actual_speedup', ''))
            speedup_error = safe_float(last_row.get('speedup_error', ''))
            prediction_accuracy = safe_float(last_row.get('prediction_accuracy_pct', ''))
            time_saved = full_circuit_time - circuit_cutting_time if full_circuit_time > 0 else 0

            # Check if full circuit simulation was run
            full_circuit_was_run = (full_circuit_time > 0)

            # Extract timing components from CSV
            find_cuts_time = safe_float(last_row.get('find_cuts_time', ''))
            cutting_transpile = safe_float(last_row.get('circuit_cutting_transpile_time_secs', ''))
            cutting_reconstruct = safe_float(last_row.get('circuit_cutting_reconstruct_time_secs', ''))
            full_transpile = safe_float(last_row.get('full_circuit_transpile_time_secs', ''))
            full_exec = safe_float(last_row.get('full_circuit_exec_time_secs', ''))

            # Print consolidated results
            print("\n" + "=" * 80)
            if full_circuit_was_run:
                print("PERFORMANCE COMPARISON: QDreamer Prediction vs Actual Execution")
                print("=" * 80)
                print(f"Speedup Metrics:")
                print(f"  Predicted Speedup:    {predicted_speedup:.3f}x")
                print(f"  Actual Speedup:       {actual_speedup:.3f}x  (measured: {circuit_cutting_time:.3f}s vs {full_circuit_time:.3f}s)")
                print(f"  Prediction Accuracy:  {prediction_accuracy:.1f}%  (error: {speedup_error:.3f}x)")
                print(f"  Time Saved:           {time_saved:.3f}s")
                print()
                print(f"Timing Breakdown:")
                print(f"  Circuit Cutting:  {circuit_cutting_time:.3f}s  (find: {find_cuts_time:.3f}s, transpile: {cutting_transpile:.3f}s, exec: {cutting_exec:.3f}s, reconstruct: {cutting_reconstruct:.3f}s)")
                print(f"    └─ Tasks: {num_tasks} subcircuits @ {avg_task_time:.3f}s avg")
                print(f"  Full Circuit:     {full_circuit_time:.3f}s  (transpile: {full_transpile:.3f}s, exec: {full_exec:.3f}s)")
            else:
                print("CIRCUIT CUTTING EXECUTION SUMMARY")
                print("=" * 80)
                print(f"Timing Breakdown:")
                print(f"  Circuit Cutting:  {circuit_cutting_time:.3f}s  (find: {find_cuts_time:.3f}s, transpile: {cutting_transpile:.3f}s, exec: {cutting_exec:.3f}s, reconstruct: {cutting_reconstruct:.3f}s)")
                print(f"    └─ Tasks: {num_tasks} subcircuits @ {avg_task_time:.3f}s avg")
                print(f"  Note: Full circuit simulation was not run (circuit_cutting_only mode)")
            print("=" * 80)

            print(f"\n✓ All metrics (predictions + measurements) saved in unified CSV:")
            print(f"  {result_file}")

            # Return summary for main() to collect
            return {
                'num_qubits': num_qubits,
                'subcircuit_size': circuit_cutting_config['subcircuit_size'],
                'num_cuts': safe_int(last_row.get('num_cuts', '')),
                'predicted_speedup': predicted_speedup,
                'actual_speedup': actual_speedup,
                'speedup_error': speedup_error,
                'prediction_accuracy': prediction_accuracy,
                'circuit_cutting_time': circuit_cutting_time,
                'full_circuit_time': full_circuit_time,
                'num_tasks': num_tasks,
                'success': True
            }

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80 + "\n")


def run_callibration(
    num_qubits, 
    executor, 
    result_file, 
    use_gpu=False, 
    gpu_fraction=0.25,
    subcircuit_sizes=None,
    min_subcircuit_size=None,
    max_subcircuit_size=None,
    num_repeats=1,
    num_nodes=1,
    cores_per_node=224,
    gpus_per_node=4,
    num_samples=10000,
    circuit_type="EfficientSU2",
    circuit_depth=2
):
    """
    Run calibration experiments with different subcircuit sizes (and thus different cut counts).
    Executes REAL circuit cutting experiments for each configuration.
    
    Args:
        num_qubits: Number of qubits in full circuit
        executor: MiniAppExecutor instance
        result_file: Path to CSV file (will append results for each scenario)
        use_gpu: Whether to use GPU
        gpu_fraction: GPU fraction per task
        subcircuit_sizes: List of specific subcircuit sizes to test (optional)
        min_subcircuit_size: Minimum subcircuit size (optional)
        max_subcircuit_size: Maximum subcircuit size (optional)
        num_repeats: Number of times to repeat each subcircuit size configuration (default: 1)
        num_nodes: Number of nodes in the cluster (for CSV tracking)
        cores_per_node: Number of CPU cores per node (for CSV tracking)
        gpus_per_node: Number of GPUs per node (for CSV tracking)
        num_samples: Number of samples for circuit cutting (default: 10000)
        circuit_type: Type of circuit to use (default: "EfficientSU2")
        circuit_depth: Circuit depth/reps (default: 2)
    
    Returns:
        List of result dictionaries, one per scenario (includes all repeats)
    """
    # Create circuit (same for all scenarios)
    circuit = EfficientSU2(num_qubits, entanglement='linear', reps=circuit_depth).decompose()
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
    # Store EfficientSU2 metadata for circuit type detection in motif.py
    circuit._efficientsu2_reps = circuit_depth
    
    # Determine subcircuit sizes to test
    if subcircuit_sizes is None:
        # Auto-generate range
        if min_subcircuit_size is None:
            min_subcircuit_size = max(10, num_qubits - 12)  # At least 2 cuts
        if max_subcircuit_size is None:
            max_subcircuit_size = num_qubits - 1  # At least 1 cut
        
        # Generate range (test every 2-3 qubits for efficiency)
        step = max(1, (max_subcircuit_size - min_subcircuit_size) // 8)
        subcircuit_sizes = list(range(min_subcircuit_size, max_subcircuit_size + 1, step))
    
    print(f"\n{'='*80}")
    print(f"CALIBRATION SWEEP: {num_qubits} qubits {'(GPU Mode)' if use_gpu else '(CPU Mode)'}")
    print(f"Circuit: {circuit_type}, depth/reps: {circuit_depth}")
    print(f"Testing {len(subcircuit_sizes)} subcircuit sizes: {subcircuit_sizes}")
    print(f"Repeats per subcircuit size: {num_repeats}")
    print(f"Total experiments: {len(subcircuit_sizes)} × {num_repeats} = {len(subcircuit_sizes) * num_repeats}")
    print(f"Results will be appended to: {result_file}")
    print(f"{'='*80}")
    
    # Detect GPU type if using GPU
    gpu_type = detect_gpu_type() if use_gpu else None
    if use_gpu:
        print(f"Detected GPU type: {gpu_type}")
    
    results = []
    
    # Initialize QDreamer once (for getting predictions)
    qdreamer = QDreamerCircuitCutting(executor=executor, 
                                circuit=circuit, 
                                use_gpu=use_gpu)
    
    # Get task resources and backend options (same for all scenarios)
    (sub_circuit_task_resources, full_circuit_task_resources,
     circuit_cutting_backend_opts, full_circuit_backend_opts) = get_task_resources_and_backend_opts(
        use_gpu=use_gpu, gpu_fraction=gpu_fraction
    )
    
    # Outer loop: repeat entire sweep num_repeats times for better statistical distribution
    for repeat in range(1, num_repeats + 1):
        if num_repeats > 1:
            print(f"\n{'='*80}")
            print(f"REPEAT {repeat}/{num_repeats}")
            print(f"{'='*80}")
        
        for i, subcircuit_size in enumerate(subcircuit_sizes):
            if num_repeats > 1:
                print(f"\n{'='*80}")
                print(f"[Repeat {repeat}/{num_repeats}] [{i+1}/{len(subcircuit_sizes)}] Testing subcircuit_size={subcircuit_size}q")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print(f"[{i+1}/{len(subcircuit_sizes)}] Testing subcircuit_size={subcircuit_size}q")
                print(f"{'='*80}")
            
            # Get QDreamer prediction for this subcircuit size
            allocation = qdreamer.evaluate_subcircuit_size(subcircuit_size)
            
            print(f"  Configuration: {subcircuit_size}q subcircuits")
            print(f"  Cuts: {allocation.num_cuts}")
            print(f"  Subexperiments: {allocation.num_parallel_tasks}")
            print(f"  Predicted speedup: {allocation.speedup_factor:.3f}x")
            print(f"  Sampling overhead: {allocation.sampling_overhead:.2f}x")
            
            # NOW RUN THE ACTUAL EXPERIMENT
            print(f"\n  Running actual circuit cutting experiment...")
            
            try:
                # Build CircuitCutting motif with this specific subcircuit size
                cc_builder = CircuitCuttingBuilder()
                cc = (
                    cc_builder
                    .set_subcircuit_size(subcircuit_size)  # Use explicit subcircuit size
                    .set_base_qubits(num_qubits)
                    .set_observables(["Z" + "I" * (num_qubits - 1)])
                    .set_scale_factor(1)
                    .set_num_samples(num_samples)
                    .set_sub_circuit_task_resources(sub_circuit_task_resources)
                    .set_full_circuit_task_resources(full_circuit_task_resources)
                    .set_result_file(result_file)  # Same file, will append
                    .set_circuit_cutting_qiskit_options({
                        "backend_options": circuit_cutting_backend_opts,
                        "mpi": False
                    })
                    .set_full_circuit_qiskit_options({
                        "backend_options": full_circuit_backend_opts,
                        "mpi": False
                    })
                    .set_full_circuit_only(False)  
                    .set_circuit_cutting_only(False)
                    .set_scenario_label(f"calibration_{num_qubits}q_{subcircuit_size}q_{allocation.num_cuts}cuts_repeat{repeat}")
                    .set_qdreamer_allocation(allocation)  # Pass predictions for comparison
                    .set_gpu_mode(use_gpu)
                    .set_gpu_fraction(gpu_fraction)
                    .set_gpu_type(gpu_type)  # Track GPU type in CSV
                    .set_circuit(circuit)  
                    .set_num_nodes(num_nodes)  # Track parallelization config in CSV
                    .set_cores_per_node(cores_per_node)
                    .set_gpus_per_node(gpus_per_node)
                    .build(executor)
                )
                
                # Execute the actual circuit cutting experiment
                with cc:
                    cc.run()
                
                print(f"  ✓ Experiment completed successfully!")
                
                # Read results from CSV (last row should be this experiment)
                with open(result_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        
                        # Extract actual measurements
                        circuit_cutting_time = safe_float(last_row.get('circuit_cutting_total_runtime_secs', ''))
                        full_circuit_time = safe_float(last_row.get('full_circuit_total_runtime_secs', ''))
                        actual_speedup = safe_float(last_row.get('actual_speedup', ''))
                        speedup_error = safe_float(last_row.get('speedup_error', ''))
                        prediction_accuracy = safe_float(last_row.get('prediction_accuracy_pct', ''))
                        num_tasks = safe_int(last_row.get('number_of_tasks', ''))
                        
                        print(f"  Results:")
                        print(f"    Actual speedup: {actual_speedup:.3f}x")
                        print(f"    Prediction accuracy: {prediction_accuracy:.1f}%")
                        print(f"    Circuit cutting time: {circuit_cutting_time:.3f}s")
                        print(f"    Full circuit time: {full_circuit_time:.3f}s")
                        
                        results.append({
                            'num_qubits': num_qubits,
                            'subcircuit_size': subcircuit_size,
                            'num_cuts': allocation.num_cuts,
                            'num_subexperiments': allocation.num_parallel_tasks,
                            'predicted_speedup': allocation.speedup_factor,
                            'actual_speedup': actual_speedup,
                            'speedup_error': speedup_error,
                            'prediction_accuracy': prediction_accuracy,
                            'circuit_cutting_time': circuit_cutting_time,
                            'full_circuit_time': full_circuit_time,
                            'num_tasks': num_tasks,
                            'success': True
                        })
                    else:
                        print(f"  ⚠ Warning: No results found in CSV")
                        results.append({
                            'num_qubits': num_qubits,
                            'subcircuit_size': subcircuit_size,
                            'num_cuts': allocation.num_cuts,
                            'success': False,
                            'error': 'No CSV results'
                        })
            
            except Exception as e:
                print(f"  ✗ Experiment failed: {e}")
                logging.exception(f"Error in calibration experiment for {subcircuit_size}q subcircuits")
                results.append({
                    'num_qubits': num_qubits,
                    'subcircuit_size': subcircuit_size,
                    'num_cuts': allocation.num_cuts,
                    'success': False,
                    'error': str(e)
                })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"CALIBRATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total scenarios tested: {len(subcircuit_sizes)}")
    print(f"Repeats per scenario: {num_repeats}")
    print(f"Total experiments: {len(subcircuit_sizes) * num_repeats}")
    print(f"Successful: {sum(1 for r in results if r.get('success'))}")
    print(f"Failed: {sum(1 for r in results if not r.get('success'))}")
    print(f"\nResults by scenario:")
    print(f"{'Subcircuit':<12} {'Cuts':<6} {'Pred Speedup':<14} {'Actual Speedup':<14} {'Accuracy':<10}")
    print(f"{'-'*60}")
    for r in results:
        if r.get('success'):
            print(f"{r['subcircuit_size']:<12} {r['num_cuts']:<6} "
                  f"{r['predicted_speedup']:<14.3f} {r['actual_speedup']:<14.3f} "
                  f"{r['prediction_accuracy']:<10.1f}")
        else:
            print(f"{r['subcircuit_size']:<12} {r['num_cuts']:<6} FAILED: {r.get('error', 'Unknown')}")
    print(f"{'='*80}")
    
    return results


def print_banner(title, char="=", width=80):
    """Print a banner with title."""
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def print_section(title, char="-", width=80):
    """Print a section header."""
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def run_calibration_experiments(config, executor, result_file, qubit_range, 
                                 execution_modes, gpu_fraction, subcircuit_sizes, 
                                 num_repeats, num_nodes, cores_per_node, gpus_per_node,
                                 num_samples, circuit_type="EfficientSU2", circuit_depth=2):
    """
    Run calibration experiments for a given parallelization config.
    
    Returns:
        tuple: (all_results, all_failed) for this config
    """
    config_label = config['label']
    all_results = []
    all_failed = []
    
    for gpu_mode in execution_modes:
        if gpu_mode and gpus_per_node == 0:
            print(f"\n  Skipping GPU mode for {config_label} (no GPUs)")
            continue

        mode_name = "GPU" if gpu_mode else "CPU"
        print_section(f"{mode_name} MODE - {config_label}")
        
        results = []
        failed = []
        
        for num_qubits in qubit_range:
            print(f"\n  Calibration: {num_qubits}q ({mode_name}, {config_label})")
            
            try:
                sweep_results = run_callibration(
                    num_qubits=num_qubits,
                    executor=executor,
                    result_file=result_file,
                    use_gpu=gpu_mode,
                    gpu_fraction=gpu_fraction,
                    subcircuit_sizes=subcircuit_sizes,
                    num_repeats=num_repeats,
                    num_nodes=num_nodes,
                    cores_per_node=cores_per_node,
                    gpus_per_node=gpus_per_node,
                    num_samples=num_samples,
                    circuit_type=circuit_type,
                    circuit_depth=circuit_depth
                )
                
                for r in sweep_results:
                    if r.get('success'):
                        r['config_label'] = config_label
                        results.append(r)
                    else:
                        failed.append((num_qubits, r.get('subcircuit_size', '?'), 
                                      mode_name, config_label, r.get('error', 'Unknown')))
                
                success_count = sum(1 for r in sweep_results if r.get('success'))
                print(f"  ✓ {num_qubits}q completed: {success_count} successful")
                
            except Exception as e:
                print(f"  ✗ {num_qubits}q failed: {e}")
                logging.exception(f"Calibration error: {num_qubits}q, {mode_name}, {config_label}")
                failed.append((num_qubits, 'all', mode_name, config_label, str(e)))
        
        # Calculate totals
        if subcircuit_sizes:
            total_mode = len(qubit_range) * len(subcircuit_sizes) * num_repeats
        else:
            total_mode = len(results) + len(failed)
        
        # Store results
        all_results.append({
            'mode': mode_name,
            'config_label': config_label,
            'num_nodes': num_nodes,
            'cores_per_node': cores_per_node,
            'gpus_per_node': gpus_per_node,
            'result_file': result_file,
            'results': results,
            'failed': failed
        })
        all_failed.extend(failed)
        
        print(f"\n  Summary: {len(results)}/{total_mode} completed, {len(failed)} failed")
    
    return all_results, all_failed


def run_single_experiments(config, executor, result_file, qubit_range,
                           execution_modes, gpu_fraction, num_repeats,
                           num_nodes, cores_per_node, gpus_per_node,
                           overall_counter, total_overall, num_samples,
                           circuit_type="EfficientSU2", circuit_depth=2):
    """
    Run single optimized experiments for a given parallelization config.
    
    Returns:
        tuple: (all_results, all_failed, updated_counter) for this config
    """
    config_label = config['label']
    all_results = []
    all_failed = []
    
    for gpu_mode in execution_modes:
        if gpu_mode and gpus_per_node == 0:
            print(f"\n  Skipping GPU mode for {config_label} (no GPUs)")
            continue

        mode_name = "GPU" if gpu_mode else "CPU"
        total_mode = len(qubit_range) * num_repeats
        print_section(f"{mode_name} MODE - {config_label}")
        
        results = []
        failed = []
        
        for num_qubits in qubit_range:
            for repeat in range(1, num_repeats + 1):
                overall_counter += 1
                mode_counter = len(results) + len(failed) + 1
                
                print(f"\n  [{overall_counter}/{total_overall}] {num_qubits}q repeat {repeat} ({mode_name})")
                
                try:
                    result = run_experiment(
                        num_qubits, executor, result_file,
                        use_gpu=gpu_mode, gpu_fraction=gpu_fraction,
                        repeat=repeat,
                        num_nodes=num_nodes,
                        cores_per_node=cores_per_node,
                        gpus_per_node=gpus_per_node,
                        num_samples=num_samples,
                        circuit_type=circuit_type,
                        circuit_depth=circuit_depth
                    )
                    
                    if result and result.get('success'):
                        result['config_label'] = config_label
                        results.append(result)
                        print(f"  ✓ Completed")
                    else:
                        failed.append((num_qubits, repeat, mode_name, config_label, "Unknown error"))
                        print(f"  ✗ Failed")
                        
                except Exception as e:
                    failed.append((num_qubits, repeat, mode_name, config_label, str(e)))
                    print(f"  ✗ Failed: {e}")
                    logging.exception(f"Experiment error: {num_qubits}q, repeat {repeat}, {mode_name}, {config_label}")
        
        # Store results
        all_results.append({
            'mode': mode_name,
            'config_label': config_label,
            'num_nodes': num_nodes,
            'cores_per_node': cores_per_node,
            'gpus_per_node': gpus_per_node,
            'result_file': result_file,
            'results': results,
            'failed': failed
        })
        all_failed.extend(failed)
        
        print(f"\n  Summary: {len(results)}/{total_mode} completed, {len(failed)} failed")
    
    return all_results, all_failed, overall_counter


def print_results_summary(all_results, all_failed, calibration_mode):
    """Print the final results summary table."""
    total_successful = sum(len(m['results']) for m in all_results)
    total_failed = len(all_failed)
    
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print("=" * 80)
    
    # Results table
    for mode_data in all_results:
        results = mode_data['results']
        if not results:
            continue
            
        mode_name = mode_data['mode']
        config_label = mode_data.get('config_label', 'N/A')
        
        print(f"\n{mode_name} ({config_label}) - {len(results)} experiments:")
        print("-" * 100)
        print(f"{'Config':<15} {'Qubits':<8} {'Subcirc':<8} {'Cuts':<6} {'Pred':<8} {'Actual':<8} {'Error':<8} {'Accuracy':<10}")
        print("-" * 100)
        
        for r in results:
            print(
                f"{r.get('config_label', config_label):<15} "
                f"{r['num_qubits']:<8} "
                f"{r['subcircuit_size']:<8} "
                f"{r['num_cuts']:<6} "
                f"{r['predicted_speedup']:<8.3f} "
                f"{r['actual_speedup']:<8.3f} "
                f"{r['speedup_error']:<8.3f} "
                f"{r['prediction_accuracy']:<10.1f}"
            )
        print("-" * 100)
    
    # Failed experiments
    if all_failed:
        print("\nFAILED EXPERIMENTS:")
        print("-" * 100)
        for item in all_failed:
            if len(item) == 5:
                qubits, detail, mode, cfg, err = item
                detail_label = "subcircuit" if calibration_mode else "repeat"
                print(f"  {qubits}q ({detail_label} {detail}, {mode}, {cfg}): {err}")
            else:
                print(f"  {item}")
        print("-" * 100)


def main():
    """
    Main function to run QDreamer experiments for multiple qubit sizes.
    All results are appended to a single unified CSV file for easy comparison.
    """
    # ============================================================================
    # Configuration
    # ============================================================================
    
    QUBIT_RANGE = [36]  # Range of qubit sizes to test
    NUM_REPEATS = 1
    GPU_FRACTION = 1
    NUM_SAMPLES_LIST = [1_000_000_000]  # List of sample counts to test
    
    # Circuit configuration
    CIRCUIT_TYPE = "EfficientSU2"  # Circuit type: "EfficientSU2" or "Custom"
    CIRCUIT_DEPTH = 1  # Circuit depth (reps for EfficientSU2)
    
    # Calibration mode settings
    CALIBRATION_MODE = True
    
    # 36 qubit Scenario
    #   1 cuts => minimal subcircuit size 18
    #   2 cuts => minimal subcircuit size 12
    #   3 cuts => minimal subcircuit size 9
    #   4 cuts => minimal subcircuit size 8
    #   5 cuts => minimal subcircuit size 6
    #   7 cuts => minimal subcircuit size 5
    #   8 cuts => minimal subcircuit size 4
    # CALIBRATION_SUBCIRCUIT_SIZES = [18, 12, 9, 7, 6, 5, 4]   # None = auto-generate
    # CALIBRATION_SUBCIRCUIT_SIZES = [18, 12, 9, 7, 6]   # None = 


    # 34 qubit Scenario
    #   1 cuts => minimal subcircuit size 17
    #   2 cuts => minimal subcircuit size 12
    #   3 cuts => minimal subcircuit size 9
    #   4 cuts => minimal subcircuit size 7
    #   5 cuts => minimal subcircuit size 6
    #   6 cuts => minimal subcircuit size 5
    #   7 cuts => minimal subcircuit size 4
    # CALIBRATION_SUBCIRCUIT_SIZES = [17, 12, 9, 7, 6, 5, 4]   
    CALIBRATION_SUBCIRCUIT_SIZES = [17, 12, 9, 7, 6]   
    
    #Parallelization configurations: (nodes, cores_per_node, gpus_per_node)
    # NOTE: Using Qiskit 1.x with qiskit-aer-gpu 0.15.1 for GPU support (not supported in Qiskit 2.x)
    PARALLELIZATION_CONFIGS = [
        # CPU configurations
        # (1, 1, 0),   # CPU mode: 1 core
        # (1, 2, 0),   # CPU mode: 2 cores
        # (1, 4, 0),   # CPU mode: 4 cores
        # (1, 8, 0),   # CPU mode: 8 cores
        # (1, 16, 0),   # CPU mode: 16 cores
        # (1, 32, 0),   # CPU mode: 32 cores
        # (1, 64, 0),   # CPU mode: 64 cores
        
        # (1, 224, 0),   # CPU mode: 256 cores
        # (1, 128, 0),   # CPU mode: 128 cores
        # GPU configurations (requires CUDA/GPU hardware)
        (1, 1, 1),   # GPU mode: 1 GPU
        (1, 2, 2),   # GPU mode: 2 GPUs
        (1, 4, 4),   # GPU mode: 4 GPUs
        (1, 8, 8),   # GPU mode: 8 GPUs
    ]


    # ============================================================================
    # Setup
    # ============================================================================
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"qdreamer_multi_qubit_{timestamp}.csv")
    
    # Calculate totals (USE_GPU will be set per config)
    num_subcircuits = len(CALIBRATION_SUBCIRCUIT_SIZES) if CALIBRATION_SUBCIRCUIT_SIZES else 2
    
    # Estimate total experiments (will vary based on GPU availability per config)
    total_experiments = 0
    for num_nodes, cores_per_node, gpus_per_node in PARALLELIZATION_CONFIGS:
        use_gpu = [True] if gpus_per_node > 0 else [False]
        if CALIBRATION_MODE:
            total_experiments += len(use_gpu) * len(QUBIT_RANGE) * num_subcircuits * NUM_REPEATS * len(NUM_SAMPLES_LIST)
        else:
            total_experiments += len(use_gpu) * len(QUBIT_RANGE) * NUM_REPEATS * len(NUM_SAMPLES_LIST)

    # ============================================================================
    # Print Configuration Summary
    # ============================================================================
    
    mode_str = "Calibration Sweep" if CALIBRATION_MODE else "Single Experiment"
    print_banner(f"QDreamer {mode_str} Benchmark with Parallelization Scaling")
    
    print(f"Qubit range: {list(QUBIT_RANGE)}")
    print(f"Repeats: {NUM_REPEATS}")
    print(f"Execution modes: Auto-determined per config (GPU if gpus_per_node > 0, else CPU)")
    print(f"GPU fraction: {GPU_FRACTION} ({int(1/GPU_FRACTION)} tasks/GPU)")
    print(f"Num samples to test: {NUM_SAMPLES_LIST}")
    
    print(f"Circuit type: {CIRCUIT_TYPE}")
    print(f"Circuit depth (reps): {CIRCUIT_DEPTH}")
    if CALIBRATION_MODE and CALIBRATION_SUBCIRCUIT_SIZES:
        print(f"Subcircuit sizes: {CALIBRATION_SUBCIRCUIT_SIZES}")
    
    print(f"\nParallelization configs ({len(PARALLELIZATION_CONFIGS)}):")
    for i, (n, c, g) in enumerate(PARALLELIZATION_CONFIGS):
        print(f"  [{i+1:2d}] nodes={n}, cores={c:3d}, gpus={g}")
    
    print(f"\nTotal experiments: ~{total_experiments}")
    print(f"Results file: {result_file}")
    print("=" * 80)

    # ============================================================================
    # Run Experiments
    # ============================================================================
    
    all_results = []
    all_failed = []
    overall_counter = 0
    
    for idx, (num_nodes, cores_per_node, gpus_per_node) in enumerate(PARALLELIZATION_CONFIGS):
        config_label = f"N{num_nodes}_C{cores_per_node}_G{gpus_per_node}"
        config = {'label': config_label, 'nodes': num_nodes, 
                  'cores': cores_per_node, 'gpus': gpus_per_node}
        
        # Set USE_GPU based on GPU availability: if gpus_per_node > 0, use GPU mode
        USE_GPU = [True] if gpus_per_node > 0 else [False]
        
        print_banner(f"CONFIG [{idx+1}/{len(PARALLELIZATION_CONFIGS)}]: {config_label}")
        print(f"  GPU mode: {'Enabled' if gpus_per_node > 0 else 'Disabled'} (gpus_per_node={gpus_per_node})")
        
        # Initialize executor
        print(f"Initializing executor...")
        executor = create_executor(num_nodes, cores_per_node, gpus_per_node)
        print(f"✓ Executor ready")
        
        try:
            # Loop over different NUM_SAMPLES values
            for sample_idx, num_samples in enumerate(NUM_SAMPLES_LIST):
                print_banner(f"NUM_SAMPLES [{sample_idx+1}/{len(NUM_SAMPLES_LIST)}]: {num_samples}")
                
                if CALIBRATION_MODE:
                    results, failed = run_calibration_experiments(
                        config, executor, result_file, QUBIT_RANGE,
                        USE_GPU, GPU_FRACTION, CALIBRATION_SUBCIRCUIT_SIZES,
                        NUM_REPEATS, num_nodes, cores_per_node, gpus_per_node,
                        num_samples, CIRCUIT_TYPE, CIRCUIT_DEPTH
                    )
                else:
                    results, failed, overall_counter = run_single_experiments(
                        config, executor, result_file, QUBIT_RANGE,
                        USE_GPU, GPU_FRACTION, NUM_REPEATS,
                        num_nodes, cores_per_node, gpus_per_node,
                        overall_counter, total_experiments, num_samples,
                        CIRCUIT_TYPE, CIRCUIT_DEPTH
                    )
                
                all_results.extend(results)
                all_failed.extend(failed)
            
        finally:
            print(f"\nClosing executor...")
            executor.close()
            print(f"✓ Executor closed")

    # ============================================================================
    # Final Summary
    # ============================================================================
    
    print_banner("BENCHMARK SUMMARY")
    print(f"Configs tested: {len(PARALLELIZATION_CONFIGS)}")
    print(f"Execution modes: Auto-determined per config (GPU if gpus_per_node > 0, else CPU)")
    print(f"Qubit sizes: {list(QUBIT_RANGE)}")
    if CALIBRATION_MODE:
        print(f"Mode: Calibration Sweep")
    else:
        print(f"Repeats per size: {NUM_REPEATS}")
    
    print_results_summary(all_results, all_failed, CALIBRATION_MODE)
    
    print(f"\n✓ Results saved to: {result_file}")
    print_banner("Benchmark completed!")


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()
