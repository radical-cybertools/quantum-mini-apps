import collections
import os
import time
from time import sleep
import copy
import csv

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.library import XGate, YGate, ZGate, HGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit.visualization import circuit_drawer
from qiskit_aer.noise import NoiseModel, phase_damping_error, thermal_relaxation_error
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService

RESOURCE_URL_HPC = "ssh://localhost"
WORKING_DIRECTORY = os.path.join(os.environ["HOME"], "work")

pilot_compute_description_ray = {
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

DD_SEQUENCES = {
    "XY4": [XGate(), YGate(), XGate(), YGate()],
}

def start_pilot(pilot_compute_description_ray):
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
    pcd = pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)
    pcd.wait()
    time.sleep(60)
    return pcs

def create_circuit_and_observable(num_qubits=7):
    circuit = EfficientSU2(num_qubits, entanglement="linear", reps=2).decompose()
    print("Unique gate names in circuit:", set(instr.operation.name for instr in circuit.data))
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)    
    
    # Create an observable based on the number of qubits
    pauli_strings = []
    for i in range(min(3, num_qubits)):  # Use at most 3 observables
        pauli_str = ['I'] * num_qubits
        pauli_str[i] = 'Z'
        pauli_strings.append(''.join(pauli_str))

    observable = SparsePauliOp(pauli_strings)
    print(f"Created observable: {pauli_strings}")
    
    return circuit, observable

def get_noise_model(noise_strength=0.05):
    noise_model = NoiseModel()
    phase_error = phase_damping_error(noise_strength)  # Adjust probability as needed

    # Apply to all common 1-qubit gates
    noisy_gates = ['x', 'y', 'z', 'h', 'sx', 'rz', 'u1', 'u2', 'u3']
    noise_model.add_all_qubit_quantum_error(phase_error, noisy_gates)

    return noise_model

def apply_dd(circuit, dd_sequence_type="XY4", logger=None):
    if dd_sequence_type not in DD_SEQUENCES:
        raise ValueError(f"Unknown DD sequence type: {dd_sequence_type}. Choose from {list(DD_SEQUENCES.keys())}")

    dd_sequence = DD_SEQUENCES[dd_sequence_type]

    num_qubits = circuit.num_qubits
    new_circuit = QuantumCircuit(circuit.qubits)

    last_gate_layer = [-1] * num_qubits
    current_layer = 0

    for instr_tuple in circuit.data:
        instr = instr_tuple.operation
        qargs = instr_tuple.qubits
        cargs = instr_tuple.clbits

        involved_qubits = [circuit.find_bit(q).index for q in qargs]

        for q in range(num_qubits):
            if q in involved_qubits:
                idle_time = current_layer - last_gate_layer[q]
                if last_gate_layer[q] != -1 and idle_time > 1: #aanpassen naar 2
                    print(f"[DD] Inserting {dd_sequence_type} on qubit {q} at layer {current_layer} (idle for {idle_time} layers)")
                    for gate in dd_sequence:
                        new_circuit.append(gate, [q])
                    new_circuit.barrier(q)
                last_gate_layer[q] = current_layer

        new_circuit.append(instr, qargs, cargs)
        current_layer += 1

    # Optional final DD
    for q in range(num_qubits):
        idle_time = current_layer - last_gate_layer[q]
        if last_gate_layer[q] != -1 and idle_time > 1:
            print(f"[DD] Final DD on qubit {q} at end (idle for {idle_time} layers)")
            for gate in dd_sequence:
                new_circuit.append(gate, [q])
            new_circuit.barrier(q)

    return new_circuit

def run_noisy_circuit(observable, backend_options, circuit, noise_model):
    backend_opts = backend_options.copy()
    backend = AerSimulator(noise_model=noise_model, **backend_opts["backend_options"])
    estimator = EstimatorV2(options={"backend_options": backend.options})
    result = estimator.run([(circuit, observable)]).result()
    return result[0].data.evs

def analyze_noise_model(noise_model):
    print("\nNOISE MODEL ANALYSIS")
    print(f"Basis gates: {noise_model.basis_gates}")
    print(f"Instructions with noise: {noise_model.noise_instructions}")
    
    # Describe the noise model based on how it was created
    print("\nNoise model description (based on creation function):")
    print("- Phase damping error on idle gates (probability 0.05)")

def write_dd_results_to_csv(results, filename="dd_effectiveness_results.csv"):
    if not results:
        print("No DD results to write.")
        return

    fieldnames = list(results[0].keys())
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"DD effectiveness results written to: {filename}")

if __name__ == "__main__":
    pcs = None
    num_nodes = [1]
    for nodes in num_nodes:
        start_time = time.time()
        try:
            # Start Pilot
            pilot_compute_description_ray["number_of_nodes"] = nodes
            pcs = start_pilot(pilot_compute_description_ray)
            logger = pcs.get_logger()
            
            # Create circuit and observable
            circuit, observable = create_circuit_and_observable(num_qubits=7)
            
            # Define backend options
            backend_options = {
                "backend_options": {
                    "shots": 4096,
                    "device": "CPU",
                    "method": "density_matrix",  # more realistic noise propagation than statevector
                    "blocking_enable": True,
                    "batched_shots_gpu": True,
                    "blocking_qubits": 25
                }
            }

            # Create noise model
            noise_model = get_noise_model()
            backend = AerSimulator(noise_model=noise_model, **backend_options["backend_options"])

            print("\nUsing the following noise model:\n")
            print(noise_model)
            
            # Analyze the noise model parameters
            analyze_noise_model(noise_model)
            
            # Transpile the circuit
            print("*********************************** Transpiling circuit ***********************************")
            pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
            transpiled_circuit = pass_manager.run(circuit)
            print("*********************************** Transpiling done ***********************************")
            
            # Print and draw the transpiled circuit
            print("\n--- Transpiled circuit ---")
            print(transpiled_circuit.draw())
            try:
                circuit_drawer(transpiled_circuit, output='mpl').show()
            except:
                pass  # Ignore if running headless
            
            # Apply DD
            dd_sequence_type = "XY4"
            dd_circuit = apply_dd(transpiled_circuit, dd_sequence_type, logger)
            
            # Print and draw the DD circuit
            print("\n--- Circuit with DD ---")
            print(dd_circuit.draw())
            try:
                circuit_drawer(dd_circuit, output='mpl').show()
            except:
                pass  # Ignore if running headless
            
            # Print depth info
            print(f"Original circuit depth: {transpiled_circuit.depth()}")
            print(f"Circuit depth after applying {dd_sequence_type} DD: {dd_circuit.depth()}")
            
            # Run noisy circuit without DD
            print("\n*********************************** Running noisy circuit without DD ***********************************")
            noisy_task = pcs.submit_task(
                run_noisy_circuit, 
                observable, 
                backend_options, 
                transpiled_circuit,
                noise_model,
                resources={'num_cpus': 1, 'num_gpus': 2, 'memory': None}
            )
            
            # Run noisy circuit with DD
            print("\n*********************************** Running noisy circuit with DD ***********************************")
            dd_noisy_task = pcs.submit_task(
                run_noisy_circuit, 
                observable, 
                backend_options, 
                dd_circuit,
                noise_model,
                resources={'num_cpus': 1, 'num_gpus': 2, 'memory': None}
            )
            
            # Get results
            results_list = pcs.get_results([noisy_task, dd_noisy_task])
            noisy_result = results_list[0]
            noisy_with_DD = results_list[1]
            
            # Print results
            print("\n----- RESULTS -----")
            print(f"Noisy result (without DD): {noisy_result}")
            print(f"Noisy result (with DD applied): {noisy_with_DD}")
            
            # Calculate DD effectiveness
            dd_results = []
            if noisy_result is not None and noisy_with_DD is not None:
                error_diff = abs(noisy_result - noisy_with_DD)
                print(f"Difference between noisy and DD results: {error_diff}")
                
                # Record results
                result_entry = {
                    "sequence": dd_sequence_type,
                    "noisy": noisy_result,
                    "noisy_dd": noisy_with_DD,
                    "dd_effect_magnitude": error_diff,
                    "qubit_count": transpiled_circuit.num_qubits,
                    "gate_depth": transpiled_circuit.depth()
                }
                dd_results.append(result_entry)
                
                # Write results to CSV
                write_dd_results_to_csv(dd_results)
            else:
                print("Error: Failed to evaluate DD effect because one or both results are None.")
            
            # Print execution time
            end_time = time.time()
            print(f"Total execution time: {end_time - start_time} seconds")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if pcs:                
                pcs.cancel()