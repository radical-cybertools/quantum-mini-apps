import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pennylane as qml
from engine.manager import MiniAppExecutor
from engine.metrics.csv_writer import MetricsFileWriter
from collections import Counter
import json

# Device selection
def get_device(n_qubits, noisy=False, shots=1024):
    if noisy:
        dev = qml.device("default.mixed", wires=n_qubits, shots=shots)
    else:
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    return dev

# Dynamical decouling implementation
def apply_dd_sequence(wires):
    for w in wires:
        qml.PauliX(w)
        qml.PauliY(w)
        qml.PauliX(w)
        qml.PauliY(w)

# Quantum circuits
def dd_copula_ansatz(params, wires, noise_prob=None):
    n_qubits = len(wires)
    depth = params.shape[0] - 1

    for wire in range(n_qubits):
        qml.RY(params[0, wire], wires=wires[wire])

    for d in range(1, depth + 1):
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[wires[i], wires[i+1]])

            # Noise check
            if noise_prob is not None and noise_prob > 0.0:
                #qml.AmplitudeDamping(noise_prob, wires=wires[i+1])
                qml.QubitChannel(get_phase_damping_kraus(noise_prob), wires=wires[i+1])
        
        apply_dd_sequence(wires)

        for wire in range(n_qubits):
            qml.RY(params[d, wire], wires=wires[wire])

def regular_copula_ansatz(params, wires, noise_prob=None):
    n_qubits = len(wires)
    depth = params.shape[0] - 1

    for wire in range(n_qubits):
        qml.RY(params[0, wire], wires=wires[wire])

    for d in range(1, depth + 1):
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[wires[i], wires[i+1]])

            # Noise check
            if noise_prob is not None and noise_prob > 0.0:
                #qml.AmplitudeDamping(noise_prob, wires=wires[i+1])
                qml.QubitChannel(get_phase_damping_kraus(noise_prob), wires=wires[i+1])
        
        for wire in range(n_qubits):
            qml.RY(params[d, wire], wires=wires[wire])       

def summarize_counts(counts):
    """Summarize counts by showing the most common bitstring and its probability."""
    if not counts:
        return "No counts"
    total = sum(counts.values())
    most_common = max(counts.items(), key=lambda x: x[1])
    prob = most_common[1] / total
    return f"Most common: {most_common[0]} (probability: {prob:.2f})"

# Noise model
def get_phase_damping_kraus(prob):
    return [
        np.array([[1, 0], [0, np.sqrt(1 - prob)]]),
        np.array([[0, 0], [0, np.sqrt(prob)]]),
    ]

def run_circuit_task(parameters, use_dd=True, use_noise=False, noise_level=0.05):
    try:
        n_qubits = parameters['n_qubits']
        circuit_depth = parameters['circuit_depth']
        
        # Create device
        dev = get_device(n_qubits, noisy=use_noise)
        
        # Generate random parameters if none provided
        if 'circuit_params' in parameters:
            params = parameters['circuit_params']
        else:
            params = np.random.uniform(0, 2*np.pi, size=(circuit_depth+1, n_qubits))
        
        # Create QNode with sampling
        if use_dd:
            @qml.qnode(dev)
            def circuit():
                dd_copula_ansatz(params, wires=range(n_qubits), noise_prob=noise_level if use_noise else None)
                return qml.sample(wires=range(n_qubits))
        else:
            @qml.qnode(dev)
            def circuit():
                regular_copula_ansatz(params, wires=range(n_qubits), noise_prob=noise_level if use_noise else None)
                return qml.sample(wires=range(n_qubits))

        # Get circuit drawing before execution (for visualization)
        circuit_drawing = qml.draw(circuit)()
        
        # Execute circuit and collect samples with timing
        start_time = time.time()
        samples = circuit()
        execution_time = time.time() - start_time
        
        # Handle shape for single vs multi-shot samples
        if samples.ndim == 1:
            samples = np.expand_dims(samples, axis=0)

        # Convert to bitstring counts
        bitstrings = [''.join(str(int(bit)) for bit in row) for row in samples]
        counts = Counter(bitstrings)
        counts_dict = dict(counts)
                
        return {
            "counts": counts_dict,
            "execution_time": execution_time,
            "circuit_drawing": circuit_drawing,
            "use_dd": use_dd,
            "use_noise": use_noise,
            "noise_level": noise_level,
            "n_qubits": n_qubits,
            "circuit_depth": circuit_depth
        }
    
    except Exception as e:
        print(f"[ERROR] Circuit execution failed: {e}")
        return {
            "error": str(e), 
            "counts": {}, 
            "execution_time": 0, 
            "circuit_drawing": "",
            "n_qubits": parameters.get('n_qubits', 0),
            "circuit_depth": parameters.get('circuit_depth', 0),
            "use_dd": use_dd,
            "use_noise": use_noise,
            "noise_level": noise_level
        }

# Experiment class
class ConductExperiment:
    def __init__(self, cluster_config):
        self.executor = MiniAppExecutor(cluster_config).get_executor() if cluster_config else None
        
        # Set up results directory
        self.current_datetime = datetime.datetime.now()
        self.timestamp = self.current_datetime.strftime('%Y-%m-%dT%H:%M:%S')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.result_dir = os.path.join(script_dir, "results")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        # Create results dataframe
        self.results_df = pd.DataFrame(columns=[
            'timestamp', 'n_qubits', 'circuit_depth', 'noise_level', 
            'use_dd', 'use_noise', 'circuit_type', 'run_number', 
            'execution_time', 'compute_time', "bitstring_counts"
        ])
    
    def run_single_experiment(self, parameters, use_dd, use_noise, noise_level, run_number):
        start_time = time.time()
        
        if self.executor:
            future = self.executor.submit_task(
                run_circuit_task, 
                parameters, 
                use_dd=use_dd, 
                use_noise=use_noise,
                noise_level=noise_level
            )
            result_dict = self.executor.get_results([future])[0]
        else:
            print(f"[ERROR] Circuit execution failed")
            
        compute_time = time.time() - start_time
        
        # Extract bitstring counts from the result
        counts = result_dict.get('counts', {})
        
        # Convert counts to JSON string for CSV
        counts_str = json.dumps(counts)
        
        # Set circuit type
        if not use_noise:
            circuit_type = "Ideal (No Noise, No DD)"
        elif use_noise and not use_dd:
            circuit_type = "Noisy (No DD)"
        else:
            circuit_type = "Noisy + DD"
            
        # Add result to dataframe
        new_row = {
            'timestamp': self.timestamp,
            'n_qubits': parameters['n_qubits'],
            'circuit_depth': parameters['circuit_depth'],
            'noise_level': noise_level,
            'use_dd': use_dd,
            'use_noise': use_noise,
            'circuit_type': circuit_type,
            'run_number': run_number,
            'execution_time': result_dict.get('execution_time', 0),
            'compute_time': compute_time,
            "bitstring_counts": counts_str
        }
        
        self.results_df = pd.concat([self.results_df, pd.DataFrame([new_row])], ignore_index=True)
        
        return result_dict
    
    def run_parameter_sweep(self, qubits_range, depths_range, noise_levels, runs_per_config=3):
        print(f"Parameter sweep")
        print(f"Qubit range: {qubits_range}")
        print(f"Depth range: {depths_range}")
        print(f"Noise levels: {noise_levels}")
        print(f"Runs per configuration: {runs_per_config}")
        
        total_configs = len(qubits_range) * len(depths_range) * len(noise_levels) * 3 * runs_per_config
        print(f"Total experiments to run: {total_configs}")
        
        experiment_count = 0
        
        # Sweep through all parameter combinations
        for n_qubits in qubits_range:
            for circuit_depth in depths_range:
                # Generate random parameters once for each qubit/depth combination
                circuit_params = np.random.uniform(0, 2*np.pi, size=(circuit_depth+1, n_qubits))
                parameters = {
                    'n_qubits': n_qubits,
                    'circuit_depth': circuit_depth,
                    'circuit_params': circuit_params  # Use same params for fair comparison
                }
                
                for noise_level in noise_levels:
                    print(f"\n{'='*60}")
                    print(f"CONFIGURATION: {n_qubits} qubits, depth {circuit_depth}, noise {noise_level}")
                    print(f"{'='*60}")
                    
                    # For each parameter set, run all three circuit types
                    for (use_dd, use_noise) in [(False, False), (False, True), (True, True)]:
                        # Get circuit type for display
                        if not use_noise:
                            circuit_type = "Ideal (No Noise, No DD)"
                        elif use_noise and not use_dd:
                            circuit_type = "Noisy (No DD)"
                        else:
                            circuit_type = "Noisy + DD"
                        
                        # Run multiple times per configuration
                        for run in range(1, runs_per_config + 1):
                            experiment_count += 1
                            progress = (experiment_count / total_configs) * 100
                            
                            print(f"\nRunning {circuit_type} - Run {run}/{runs_per_config}")
                            print(f"Progress: {experiment_count}/{total_configs} ({progress:.1f}%)")
                            
                            result = self.run_single_experiment(
                                parameters, 
                                use_dd=use_dd, 
                                use_noise=use_noise, 
                                noise_level=noise_level, 
                                run_number=run
                            )
                            
                            # Print an example of the bitstring counts to verify they're being collected
                            if 'counts' in result and result['counts']:
                                top_count = max(result['counts'].items(), key=lambda x: x[1]) if result['counts'] else None
                                print(f"  Top bitstring: {top_count[0]} (count: {top_count[1]})") if top_count else print("  No bitstrings found!")
                            else:
                                print("  No counts found in result!")
                                
                            print(f"  Execution time: {result.get('execution_time', 0):.6f} seconds")
                            
        # Save results to CSV
        csv_path = os.path.join(self.result_dir, f"results_{self.timestamp}.csv")
        self.results_df.to_csv(csv_path, index=False)
        print(f"\n[INFO] All results saved to {csv_path}")
        
        return self.results_df

# Main execution 
if __name__ == "__main__":
    RESOURCE_URL_HPC = "ssh://localhost"
    WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")
    
    cluster_info = {       
        "executor": "pilot",
        "config": {
            "resource": RESOURCE_URL_HPC,
            "working_directory": WORKING_DIRECTORY,
            "number_of_nodes": 2,
            "cores_per_node": 8,
            "gpus_per_node": 2,
            "queue": "debug",
            "walltime": 30,
            "type": "ray",
            "scheduler_script_commands": ["#SBATCH --partition=gpua16", "#SBATCH --gres=gpu:2"]
        }
    }
    
    try:
        print("Dynamic decoupling pennylane experiment")
        print("This experiment will test:")
        print("  - Different numbers of qubits")
        print("  - Different circuit depths")
        print("  - Different noise levels")
        print("  - With and without dynamical decoupling (DD)")
        print("Results will be averaged over multiple runs for accuracy.\n")
        
        # Create experiment
        experiment = ConductExperiment(cluster_info)
        
        # Parameter ranges to test
        qubit_range = [3, 4, 5, 6, 7, 8, 9, 10]  
        depth_range = [1, 2, 3]     
        noise_levels = [0.05]
        runs_per_config = 3
        
        # Run the parameter sweep
        df = experiment.run_parameter_sweep(
            qubits_range=qubit_range,
            depths_range=depth_range,
            noise_levels=noise_levels,
            runs_per_config=runs_per_config
        )
        
        print("\nExperiment complete.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()