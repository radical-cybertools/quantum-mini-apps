"""
Hardware Calibration Module

This module provides functions to calibrate hardware-specific parameters for the
speedup prediction model. Calibration is done by running benchmark circuits and
fitting model parameters to actual measurements.

Usage:
    1. Run calibration to create a hardware profile:
        results = run_experiment(...)
        calibration = calibrate_from_results(results, num_qubits=25)
        save_calibration(calibration, "my_laptop_profile.json")
    
    2. Load calibration for predictions:
        calibration = load_calibration("my_laptop_profile.json")
        predictor = SpeedupPredictor(calibration=calibration)
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from mini_apps.quantum_simulation.cut_estimator import HardwareCalibration


def calibrate_from_measurements(
    measurements: List[Dict[str, Any]],
    num_qubits: int,
    full_circuit_time: Optional[float] = None
) -> HardwareCalibration:
    """
    Calibrate hardware-specific parameters from measurements.
    
    This function profiles the hardware to determine:
    - c_h: Framework overhead constant (how fast hardware executes subcircuits)
    - e_h: Parallel efficiency factor (hardware-specific scaling characteristics)
    - r_h: Reconstruction overhead constant (how fast hardware does reconstruction)
    
    Calibration focuses on hardware characteristics, not parallelization.
    The number of workers (W) is a prediction parameter, not a calibration parameter.
    
    Args:
        measurements: List of measurement dictionaries, each containing:
            - n_sub: subcircuit size
            - M: number of subexperiments
            - k: number of cuts
            - execution_time: actual execution time (for sequential execution, W=1)
            - reconstruction_time: actual reconstruction time
        num_qubits: Total number of qubits in the circuit (n)
        full_circuit_time: Full circuit execution time for normalization
    
    Returns:
        Calibrated HardwareCalibration object
    """
    if not measurements:
        from mini_apps.quantum_simulation.cut_estimator import get_default_cpu_calibration
        return get_default_cpu_calibration()
    
    if not full_circuit_time or full_circuit_time <= 0:
        raise ValueError("full_circuit_time is required for calibration")
    
    # Fit hardware characteristics
    c_h_values = []
    e_h_values = []
    r_h_values = []
    
    sampling_base = 9.0  # Typical for gate cuts
    
    for meas in measurements:
        n_sub = meas['n_sub']
        M = meas['M']  # Number of subcircuits (parallel units)
        M_seq = meas.get('M_seq', None)  # Max subexperiments per subcircuit (sequential)
        k = meas['k']
        execution_time = meas['execution_time']  # Sequential execution (W=1)
        reconstruction_time = meas['reconstruction_time']
        
        # Normalize times to be relative (normalized to full circuit time)
        execution_time_relative = execution_time / full_circuit_time
        reconstruction_time_relative = reconstruction_time / full_circuit_time
        
        # Fit c_h: Framework overhead constant
        # t_sub = 2^(n_sub - n) * c_h (relative time per subexperiment)
        # If M_seq is provided: t_par = M * M_seq * t_sub / e_h (for W=1, batches=M)
        # Otherwise (legacy): t_par = M * t_sub / e_h
        if M > 0 and execution_time_relative > 0:
            if M_seq is not None:
                # Model: subcircuits run in parallel, subexperiments within each run sequentially
                # For W=1, batches = M, so t_par = M * M_seq * t_sub / e_h
                # Assuming e_h=1 during calibration: t_sub = t_par / (M * M_seq)
                t_sub_estimate = execution_time_relative / (M * M_seq)
            else:
                # Legacy model: all subexperiments are parallel units
                # For W=1, batches = M, so t_par = M * t_sub / e_h
                # Assuming e_h=1 during calibration: t_sub = t_par / M
                t_sub_estimate = execution_time_relative / M
            
            c_h_estimate = t_sub_estimate / (2 ** (n_sub - num_qubits))
            if c_h_estimate > 0:
                c_h_values.append(c_h_estimate)
                print(f"    c_h estimate: {c_h_estimate:.2f} (t_sub_rel={t_sub_estimate:.6f}, n_sub={n_sub}, n={num_qubits}, 2^(n_sub-n)={2**(n_sub-num_qubits):.2e})")
        
        # Fit e_h: Parallel efficiency (hardware characteristic)
        # For sequential execution (W=1), e_h should be close to 1.0
        if M > 0 and execution_time_relative > 0:
            if c_h_values:
                c_h_avg = np.median(c_h_values)
                t_sub = (2 ** (n_sub - num_qubits)) * c_h_avg
                
                if M_seq is not None:
                    # t_par = M * M_seq * t_sub / e_h
                    # e_h = M * M_seq * t_sub / t_par
                    e_h_estimate = (M * M_seq * t_sub) / execution_time_relative
                else:
                    # Legacy: t_par = M * t_sub / e_h
                    # e_h = M * t_sub / t_par
                    e_h_estimate = (M * t_sub) / execution_time_relative
                
                # For sequential, e_h should be ~1.0, but hardware overhead
                if 0.5 < e_h_estimate <= 1.5:  # Allow some range
                    e_h_values.append(e_h_estimate)
        
        # Fit r_h: Reconstruction overhead (hardware characteristic)
        # t_rec = r_h * b^k (relative time)
        # r_h = t_rec / b^k
        if reconstruction_time_relative > 0 and k > 0:
            r_h_estimate = reconstruction_time_relative / (sampling_base ** k)
            if r_h_estimate > 0:
                r_h_values.append(r_h_estimate)
                print(f"    r_h estimate: {r_h_estimate:.10e} (t_rec_rel={reconstruction_time_relative:.6f}, k={k}, b^k={sampling_base**k:.2e})")
    
    # Use median to be robust to outliers
    c_h = np.median(c_h_values) if c_h_values else 1.0
    e_h = np.median(e_h_values) if e_h_values else 0.8
    r_h = np.median(r_h_values) if r_h_values else 0.001
    
    # Debug: Print raw calibration values before clamping
    if c_h_values:
        print(f"  Raw c_h values: {c_h_values}")
        print(f"  Raw c_h median: {c_h}")
    if e_h_values:
        print(f"  Raw e_h values: {e_h_values}")
        print(f"  Raw e_h median: {e_h}")
    if r_h_values:
        print(f"  Raw r_h values: {r_h_values}")
        print(f"  Raw r_h median: {r_h}")
    
    # Clamp values to reasonable ranges
    # Note: c_h can be large for small subcircuits (n_sub << n) due to 2^(n_sub-n) being very small
    # Removing the upper clamp on c_h to allow it to reflect actual hardware characteristics
    c_h = max(0.1, c_h)  # Only clamp minimum, allow large values
    e_h = max(0.1, min(1.0, e_h))
    # r_h can be very small (e.g., 1e-7) for large k values, so don't clamp minimum too high
    # Only clamp to prevent negative or zero values
    r_h = max(1e-10, min(0.1, r_h))  # Allow very small values, just prevent negative/zero
    
    return HardwareCalibration(
        c_h=float(c_h),
        e_h=float(e_h),
        r_h=float(r_h),
        s_max=100.0
    )


def calibrate_from_results(
    results: Dict,
    num_qubits: int
) -> Tuple[HardwareCalibration, Dict[str, Any]]:
    """
    Calibrate hardware parameters from experiment results.
    
    This extracts measurements from the results and calibrates
    hardware-specific parameters. Calibration should be done with
    sequential execution (W=1) to profile hardware characteristics.
    
    Usage:
        results = run_experiment(..., num_workers=1)  # Sequential for calibration
        calibrated, timing_data = calibrate_from_results(results, num_qubits=25)
        save_calibration(calibrated, "my_laptop_profile.json", metadata=timing_data)
    
    Args:
        results: Results dictionary from run_experiment()
        num_qubits: Total number of qubits in the circuit
    
    Returns:
        Tuple of (Calibrated HardwareCalibration object, timing_data dictionary)
        where timing_data contains subcircuit timing information
    """
    measurements = []
    full_circuit_time = results.get('full_circuit_time')
    
    if not full_circuit_time:
        raise ValueError("Results must include 'full_circuit_time' for calibration")
    
    # Collect subcircuit timing data
    subcircuit_timings = []
    
    for i in range(len(results['num_cuts'])):
        n_sub = results['subcircuit_sizes'][i]
        M = results['num_subexperiments'][i]
        execution_time = results['execution_times'][i]
        
        # Calculate average time per subcircuit execution
        # For sequential execution (W=1), we have M subexperiments
        # Average time per subexperiment = execution_time / M
        avg_subcircuit_time = execution_time / M if M > 0 else 0.0
        avg_subcircuit_time_relative = avg_subcircuit_time / full_circuit_time if full_circuit_time > 0 else 0.0
        
        subcircuit_timings.append({
            'subcircuit_size': n_sub,
            'num_subexperiments': M,
            'num_cuts': results['num_cuts'][i],
            'avg_subcircuit_time_seconds': avg_subcircuit_time,
            'avg_subcircuit_time_relative': avg_subcircuit_time_relative,
            'total_execution_time_seconds': execution_time,
            'total_execution_time_relative': execution_time / full_circuit_time if full_circuit_time > 0 else 0.0
        })
        
        measurements.append({
            'n_sub': n_sub,
            'M': M,
            'k': results['num_cuts'][i],
            'execution_time': execution_time,
            'reconstruction_time': results['reconstruction_times'][i],
        })
    
    calibration = calibrate_from_measurements(measurements, num_qubits, full_circuit_time)
    
    timing_data = {
        'full_circuit_time_seconds': full_circuit_time,
        'subcircuit_timings': subcircuit_timings
    }
    
    return calibration, timing_data


def save_calibration(
    calibration: HardwareCalibration,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a hardware calibration to a JSON file.
    
    Args:
        calibration: HardwareCalibration object to save
        filepath: Path to save the calibration file
        metadata: Optional metadata to include (e.g., machine name, date)
    """
    data = {
        'c_h': calibration.c_h,
        'e_h': calibration.e_h,
        'r_h': calibration.r_h,
        's_max': calibration.s_max,
    }
    
    if metadata:
        data['metadata'] = metadata
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Calibration saved to {filepath}")


def load_calibration(filepath: str) -> HardwareCalibration:
    """
    Load a hardware calibration from a JSON file.
    
    Args:
        filepath: Path to the calibration file
    
    Returns:
        HardwareCalibration object
    
    Raises:
        FileNotFoundError: If the calibration file doesn't exist
        ValueError: If the calibration file is invalid
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    try:
        calibration = HardwareCalibration(
            c_h=data['c_h'],
            e_h=data['e_h'],
            r_h=data['r_h'],
            s_max=data.get('s_max', float('inf'))
        )
        
        if 'metadata' in data:
            print(f"Loaded calibration with metadata: {data['metadata']}")
        
        return calibration
    except KeyError as e:
        raise ValueError(f"Invalid calibration file format: missing key {e}")


def list_calibrations(directory: str = ".") -> List[str]:
    """
    List all calibration files in a directory.
    
    Args:
        directory: Directory to search for calibration files
    
    Returns:
        List of calibration file paths
    """
    path = Path(directory)
    return [str(p) for p in path.glob("*_calibration.json")]


def run_calibration_benchmark(
    num_qubits: int,
    subcircuit_sizes: List[int],
    backend_options: Dict,
    num_samples: int = 1
) -> Dict[str, Any]:
    """
    Run calibration benchmark to profile hardware characteristics.
    
    This function runs:
    1. Full circuit execution (baseline)
    2. Circuit cutting with different subcircuit sizes
    3. Measures actual execution times
    
    Returns raw timing measurements for calibration.
    
    Args:
        num_qubits: Number of qubits in the circuit
        subcircuit_sizes: List of subcircuit sizes to test
        backend_options: Backend options for execution
        num_samples: Number of samples for circuit cutting
    
    Returns:
        Dictionary with timing measurements and metadata
    """
    from mini_apps.quantum_simulation.experiment_qdreamer import (
        create_circuit_and_observable,
        run_full_circuit,
        run_cut_circuit
    )
    
    print(f"Creating {num_qubits}-qubit circuit...")
    circuit, observable = create_circuit_and_observable(num_qubits, depth=3)
    
    # Run full circuit to get baseline
    print("Running full circuit simulation...")
    full_circuit_time, full_expval = run_full_circuit(circuit, observable, backend_options)
    print(f"Full circuit execution time: {full_circuit_time:.4f} seconds")
    print(f"Full circuit expectation value: {full_expval:.6f}\n")
    
    # Results storage
    measurements = {
        'full_circuit_time': full_circuit_time,
        'full_circuit_expval': full_expval,
        'subcircuit_measurements': []
    }
    
    # Test different subcircuit sizes
    for subcircuit_size in subcircuit_sizes:
        print(f"{'='*60}")
        print(f"Testing subcircuit size: {subcircuit_size}")
        print(f"{'='*60}")
        
        try:
            # Calibration uses sequential execution (num_workers=1) to profile hardware characteristics
            cut_time, num_cuts, num_subcircuits, num_subexperiments, max_subexperiments_per_subcircuit, metrics = run_cut_circuit(
                circuit, observable, subcircuit_size, backend_options, num_samples, num_workers=1
            )
            
            print(f"Found {num_cuts} cuts, {num_subcircuits} subcircuits")
            print(f"Total subexperiments: {num_subexperiments}")
            print(f"Max subexperiments per subcircuit: {max_subexperiments_per_subcircuit}")
            print(f"Execution time: {metrics['execution_time']:.4f}s")
            print(f"Reconstruction time: {metrics['reconstruction_time']:.4f}s")
            print(f"Total time: {cut_time:.4f}s")
            print(f"  - Find cuts: {metrics['find_cuts_time']:.4f}s")
            print(f"  - Transpile: {metrics['transpile_time']:.4f}s")
            print(f"  - Execute: {metrics['execution_time']:.4f}s")
            print(f"  - Reconstruct: {metrics['reconstruction_time']:.4f}s")
            
            # Calculate average time per subexperiment execution
            # Note: subexperiments run sequentially within each subcircuit
            avg_subexperiment_time = metrics['execution_time'] / num_subexperiments if num_subexperiments > 0 else 0.0
            avg_subexperiment_time_relative = avg_subexperiment_time / full_circuit_time if full_circuit_time > 0 else 0.0
            
            measurement = {
                'subcircuit_size': subcircuit_size,
                'num_cuts': num_cuts,
                'num_subcircuits': num_subcircuits,
                'num_subexperiments': num_subexperiments,
                'max_subexperiments_per_subcircuit': max_subexperiments_per_subcircuit,
                'execution_time': metrics['execution_time'],
                'reconstruction_time': metrics['reconstruction_time'],
                'find_cuts_time': metrics['find_cuts_time'],
                'transpile_time': metrics['transpile_time'],
                'total_time': cut_time,
                'avg_subexperiment_time_seconds': avg_subexperiment_time,
                'avg_subexperiment_time_relative': avg_subexperiment_time_relative,
            }
            
            measurements['subcircuit_measurements'].append(measurement)
            print()
            
        except Exception as e:
            print(f"Error testing subcircuit_size={subcircuit_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return measurements


def main():
    """
    Main function to run hardware calibration.
    
    This function profiles hardware characteristics (execution speed, efficiency)
    by running sequential experiments (W=1). The resulting calibration can then
    be used to predict speedup for any number of workers.
    
    Supports both CPU and GPU backends.
    """
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(
        description="Calibrate hardware parameters for speedup prediction model"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["CPU", "GPU"],
        default="CPU",
        help="Device to use for calibration (default: CPU)"
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=25,
        help="Number of qubits in the calibration circuit (default: 25)"
    )
    parser.add_argument(
        "--subcircuit-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Subcircuit sizes to test (default: auto-generated range)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output calibration file path (default: auto-generated)"
    )
    parser.add_argument(
        "--machine-name",
        type=str,
        default=None,
        help="Machine name for metadata (default: auto-detected)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        device_suffix = args.device.lower()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{device_suffix}_calibration.json"
    
    # Setup backend options based on device
    if args.device == "GPU":
        backend_options = {
            "device": "GPU",
            "method": "statevector",
            "blocking_enable": True,
            "batched_shots_gpu": True,
        }
        # Add blocking_qubits if available (for GPU optimization)
        if args.num_qubits > 20:
            backend_options["blocking_qubits"] = min(23, args.num_qubits - 2)
    else:
        backend_options = {
            "device": "CPU",
            "method": "statevector"
        }
    
    # Generate subcircuit sizes if not provided
    if args.subcircuit_sizes is None:
        # Try a range from 8 up to roughly half the circuit size
        max_size = min(args.num_qubits // 2, 20)
        args.subcircuit_sizes = list(range(8, max_size + 1, 2))
    
    print("="*70)
    print("Hardware Calibration - Hardware Profiling")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Circuit size: {args.num_qubits} qubits")
    print(f"Subcircuit sizes: {args.subcircuit_sizes}")
    print(f"Backend options: {backend_options}")
    print(f"Output file: {args.output}")
    print()
    print("Note: Calibration profiles hardware characteristics (execution speed).")
    print("      Sequential execution (W=1) is used for hardware profiling.")
    print("      The resulting profile can predict speedup for any number of workers.")
    print("="*70)
    print()
    
    # Run calibration benchmark (hardware profiling only)
    print("Running calibration benchmark...")
    print("This may take a while depending on circuit size and device.\n")
    
    try:
        # Run calibration benchmark - just measure execution times
        measurements = run_calibration_benchmark(
            num_qubits=args.num_qubits,
            subcircuit_sizes=args.subcircuit_sizes,
            backend_options=backend_options
        )
        
        if not measurements['subcircuit_measurements']:
            print("ERROR: No valid measurements obtained. Calibration failed.")
            return
        
        # Convert measurements to format expected by calibrate_from_measurements
        # M = number of subcircuits (parallel units)
        # M_seq = max subexperiments per subcircuit (sequential within subcircuit)
        calibration_measurements = []
        for meas in measurements['subcircuit_measurements']:
            calibration_measurements.append({
                'n_sub': meas['subcircuit_size'],
                'M': meas['num_subcircuits'],  # M = number of subcircuits (parallel units)
                'M_seq': meas['max_subexperiments_per_subcircuit'],  # Sequential subexperiments per subcircuit
                'k': meas['num_cuts'],
                'execution_time': meas['execution_time'],
                'reconstruction_time': meas['reconstruction_time'],
            })
        
        # Calibrate from measurements
        print("\n" + "="*70)
        print("Fitting calibration parameters...")
        print("="*70)
        
        calibrated = calibrate_from_measurements(
            calibration_measurements,
            num_qubits=args.num_qubits,
            full_circuit_time=measurements['full_circuit_time']
        )
        
        print(f"\nCalibrated parameters:")
        print(f"  c_h (framework overhead): {calibrated.c_h:.6f}")
        print(f"  e_h (parallel efficiency): {calibrated.e_h:.6f}")
        print(f"  r_h (reconstruction overhead): {calibrated.r_h:.8f}")
        print(f"  s_max (max speedup): {calibrated.s_max:.1f}")
        
        # Print subcircuit timing summary
        print(f"\nSubcircuit timing summary:")
        print(f"{'Subcircuit Size':<18} {'Subcircuits':<12} {'Max Subexpts/Sub':<18} {'Total Subexpts':<15} {'Avg Subexpt Time (s)':<20} {'Cuts':<8}")
        print("-" * 100)
        for meas in measurements['subcircuit_measurements']:
            print(f"{meas['subcircuit_size']:<18} "
                  f"{meas['num_subcircuits']:<12} "
                  f"{meas['max_subexperiments_per_subcircuit']:<18} "
                  f"{meas['num_subexperiments']:<15} "
                  f"{meas['avg_subexperiment_time_seconds']:<20.6f} "
                  f"{meas['num_cuts']:<8}")
        
        # Prepare metadata with all timing data
        import platform
        metadata = {
            "device": args.device,
            "num_qubits": args.num_qubits,
            "subcircuit_sizes": args.subcircuit_sizes,
            "machine": args.machine_name or platform.node(),
            "platform": platform.platform(),
            "calibration_date": datetime.datetime.now().isoformat(),
            "full_circuit_time_seconds": measurements['full_circuit_time'],
            "full_circuit_expval": measurements['full_circuit_expval'],
            "subcircuit_measurements": measurements['subcircuit_measurements']
        }
        
        # Save calibration
        save_calibration(calibrated, args.output, metadata=metadata)
        
        print("\n" + "="*70)
        print("Calibration Complete!")
        print("="*70)
        print(f"\nCalibration saved to: {args.output}")
        print("\nTo use this calibration in experiments:")
        print(f"  from mini_apps.quantum_simulation.calibration import load_calibration")
        print(f"  calibration = load_calibration('{args.output}')")
        print(f"  results = run_experiment(..., calibration=calibration)")
        
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user.")
    except Exception as e:
        print(f"\nERROR: Calibration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

