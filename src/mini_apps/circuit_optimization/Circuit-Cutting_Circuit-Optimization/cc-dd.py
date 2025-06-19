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
from qiskit_addon_cutting import (cut_wires, expand_observables,
                                  generate_cutting_experiments,
                                  partition_problem,
                                  reconstruct_expectation_values)
from qiskit_addon_cutting.automated_cut_finding import (DeviceConstraints,
                                                        OptimizationParameters,
                                                        find_cuts)
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_ibm_runtime import Batch, SamplerV2
from qiskit.primitives import (
    SamplerResult,
    PrimitiveResult,
)
from qiskit.primitives.containers import (
    DataBin,
    BitArray,
    SamplerPubResult
)
from qiskit.visualization import circuit_drawer
from qiskit_aer.noise import NoiseModel, phase_damping_error
from pilot.pilot_compute_service import ExecutionEngine, PilotComputeService

# Constants and settings
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
    "XY4": [XGate(), YGate(), XGate(), YGate()]
}

def start_pilot(pilot_compute_description_ray):
    pcs = PilotComputeService(execution_engine=ExecutionEngine.RAY, working_directory=WORKING_DIRECTORY)
    pcd = pcs.create_pilot(pilot_compute_description=pilot_compute_description_ray)
    pcd.wait()
    time.sleep(60)
    return pcs

def pre_processing(logger, num_qubits=7, qps=2, num_samples=10):       
    circuit = EfficientSU2(num_qubits, entanglement="linear", reps=2).decompose()
    # Debug: Print all gate names used
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

    # Specify settings for the cut-finding optimizer
    optimization_settings = OptimizationParameters(seed=111)

    # Specify the size of the QPUs available
    device_constraints = DeviceConstraints(qubits_per_subcircuit=qps)

    cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
    print(
        f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
        f'overhead of {metadata["sampling_overhead"]}.\n'
        f'Lowest cost solution found: {metadata["minimum_reached"]}.'
    )
    for cut in metadata["cuts"]:
        print(f"{cut[0]} at circuit instruction index {cut[1]}")

    qc_w_ancilla = cut_wires(cut_circuit)
    
    print("\n--- Cut circuit before observable expansion ---")
    try:
        circuit_drawer(qc_w_ancilla, output='mpl').show()
    except:
        pass  # Ignore if running headless
    print(qc_w_ancilla.draw())

    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)

    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    print(
        f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}"
    )

    print(f"Number of subcircuits: {len(subcircuits)}")
    # The subcircuits from partition_problem are stored in a dictionary with integer keys
    if isinstance(subcircuits, dict):
        for i, subcirc in subcircuits.items():
            print(f"\n--- Subcircuit {i} ---")
            print(f"Number of qubits: {subcirc.num_qubits}")
            print(f"Circuit depth: {subcirc.depth()}")
            print(subcirc.draw())
    else:
        # If subcircuits is already a list, iterate directly
        for i, subcirc in enumerate(subcircuits):
            print(f"\n--- Subcircuit {i} ---")
            print(f"Number of qubits: {subcirc.num_qubits}")
            print(f"Circuit depth: {subcirc.depth()}")
            print(subcirc.draw())

    # Return both the partitioned problem components and the circuit/observable for experiment generation
    return subcircuits, subobservables, partitioned_problem, observable, circuit, metadata

def apply_dd_to_subcircuits(subcircuits, dd_sequence_type="XY4", logger=None):
    if dd_sequence_type not in DD_SEQUENCES:
        raise ValueError(f"Unknown DD sequence type: {dd_sequence_type}. Choose from {list(DD_SEQUENCES.keys())}")
    
    dd_subcircuits = {}
    for i, subcircuit in subcircuits.items():
        print(f"\n--- Applying DD ({dd_sequence_type}) to subcircuit {i} ---")
        print(f"Original depth: {subcircuit.depth()}")
        dd_subcircuit = apply_dd(subcircuit, dd_sequence_type, logger)
        print(f"Depth after DD: {dd_subcircuit.depth()}")
        dd_subcircuits[i] = dd_subcircuit
        
    return dd_subcircuits

def get_dd_noise_model(noise_strength=0.05):
    noise_model = NoiseModel()
    phase_error = phase_damping_error(noise_strength)

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

def run_noiseless_circuit(observable, backend_options, circuit):
    backend_opts = backend_options.copy()
    # Create a backend without noise model
    backend = AerSimulator(**backend_opts["backend_options"])
    estimator = EstimatorV2(options={"backend_options": backend.options})
    result = estimator.run([(circuit, observable)]).result()
    return result[0].data.evs

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

# Define a custom DataBin class to fix the serialization issues
class CustomDataBin(DataBin):
    def __setattr__(self, name, value):
        # Override __setattr__ to avoid the NotImplementedError
        self.__dict__[name] = value


def execute_sampler(sampler, label, subsystem_subexpts, shots):
    print(sampler, label, subsystem_subexpts, shots)
    try:
        submit_start = time.time()
        job = sampler.run(subsystem_subexpts, shots=shots)
        submit_end = time.time()
        result_start = time.time()
        result = job.result()    
        result_end = time.time()
        print(f"Job {label} completed with job id {job.job_id()}, submit_time: {submit_end-submit_start} and execution_time: {result_end - result_start}, type: {type(result)}")
        
        total_submit_time = submit_end - submit_start
        total_exec_time = result_end - result_start
        
        # FIX: Reconstruct the PrimitiveResult object to fix serialization issues
        new_results = []
        for pub_result in result._pub_results:
            # Deep copy the metadata
            new_metadata = copy.deepcopy(pub_result.metadata)
            
            # Access the DataBin object
            data_bin = pub_result.data
            
            # Reconstruct DataBin
            new_data_bin_dict = {}
            
            # Explicitly copy 'observable_measurements'
            if hasattr(data_bin, "observable_measurements") and data_bin.observable_measurements is not None:
                observable_measurements = data_bin.observable_measurements
                new_observable_array = np.copy(observable_measurements.array)
                new_observable_bitarray = BitArray(
                    new_observable_array, observable_measurements.num_bits
                )
                new_data_bin_dict["observable_measurements"] = new_observable_bitarray
                
            # Explicitly copy 'qpd_measurements'
            if hasattr(data_bin, "qpd_measurements") and data_bin.qpd_measurements is not None:
                qpd_measurements = data_bin.qpd_measurements
                new_qpd_array = np.copy(qpd_measurements.array)
                new_qpd_bitarray = BitArray(new_qpd_array, qpd_measurements.num_bits)
                new_data_bin_dict["qpd_measurements"] = new_qpd_bitarray
                
            # Copy other attributes of DataBin (e.g., 'shape')
            if hasattr(data_bin, "shape"):
                new_data_bin_dict["shape"] = copy.deepcopy(data_bin.shape)
                
            # Create a new DataBin instance using our custom class
            new_data_bin = CustomDataBin(**new_data_bin_dict)
            
            # Create a new SamplerPubResult
            new_pub_result = SamplerPubResult(data=new_data_bin, metadata=new_metadata)
            new_results.append(new_pub_result)
            
        # Create a new PrimitiveResult
        new_result = PrimitiveResult(
            new_results, metadata=copy.deepcopy(result.metadata)
        )
        
        return (label, new_result, total_submit_time, total_exec_time)
    except Exception as e:
        print(f"Error in execute_sampler: {e}")
        raise

def pre_process_dd_first(circuit, observable, optimization_settings, device_constraints, dd_sequence_type="XY4", logger=None):
    print("\n--- Applying DD to full circuit first before cutting ---")
    print(f"Original circuit depth: {circuit.depth()}")
    full_dd_circuit = apply_dd(circuit, dd_sequence_type, logger)
    print(f"Circuit depth after DD: {full_dd_circuit.depth()}")
    
    # Now find cuts for the DD-applied circuit
    cut_circuit, metadata = find_cuts(full_dd_circuit, optimization_settings, device_constraints)
    print(
        f'Found solution using {len(metadata["cuts"])} cuts with a sampling '
        f'overhead of {metadata["sampling_overhead"]}.\n'
        f'Lowest cost solution found: {metadata["minimum_reached"]}.'
    )
    for cut in metadata["cuts"]:
        print(f"{cut[0]} at circuit instruction index {cut[1]}")

    qc_w_ancilla = cut_wires(cut_circuit)
    
    print("\n--- Cut circuit (with DD already applied) before observable expansion ---")
    try:
        circuit_drawer(qc_w_ancilla, output='mpl').show()
    except:
        pass  # Ignore if running headless
    print(qc_w_ancilla.draw())

    # Need to expand observables from the original circuit
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)

    partitioned_problem = partition_problem(
        circuit=qc_w_ancilla, observables=observables_expanded
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    print(
        f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}"
    )

    print(f"Number of subcircuits: {len(subcircuits)}")
    if isinstance(subcircuits, dict):
        for i, subcirc in subcircuits.items():
            print(f"\n--- Subcircuit {i} (DD applied before cutting) ---")
            print(f"Number of qubits: {subcirc.num_qubits}")
            print(f"Circuit depth: {subcirc.depth()}")
            print(subcirc.draw())
    else:
        for i, subcirc in enumerate(subcircuits):
            print(f"\n--- Subcircuit {i} (DD applied before cutting) ---")
            print(f"Number of qubits: {subcirc.num_qubits}")
            print(f"Circuit depth: {subcirc.depth()}")
            print(subcirc.draw())

    # Return the partitioned components
    return subcircuits, subobservables, partitioned_problem, metadata

def write_results_to_csv(results_dict, filename="circuit_cutting_dd_results.csv"):
    fieldnames = ["method", "expectation value", "description", "num_qubits", "depth", "sampling_overhead", "num_cuts"]
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_dict:
            writer.writerow(row)
    print(f"Circuit cutting and DD results written to: {filename}")

    
def write_timing_to_csv(timing_data, filename="cutting_execution_timing.csv"):
    fieldnames = ["method", "label", "submit_time", "exec_time"]
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in timing_data:
            writer.writerow(row)
    print(f"Execution timing data written to {filename}")

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
            
            # Define backend options
            backend_options = {
                "backend_options": {
                    "shots": 32768,
                    "device": "CPU",
                    "method": "density_matrix", 
                    "blocking_enable": True,
                    "batched_shots_gpu": True,
                    "blocking_qubits": 25
                }
            }

            # Add realistic noise model relevant to DD
            noise_model = get_dd_noise_model()
            backend = AerSimulator(noise_model=noise_model, **backend_options["backend_options"])
            print("\nUsing the following noise model:\n")
            print(noise_model)
            analyze_noise_model(noise_model)
            
            # 1. Run pre-processing for circuit cutting (without applying DD yet)
            print("\n*********************************** PREPROCESSING CIRCUIT ***********************************")
            subcircuits, subobservables, partitioned_problem, observable, original_circuit, metadata_no_dd = \
                pre_processing(logger, num_qubits=7, qps=2, num_samples=10)
            
            # 2. Apply DD to the individual subcircuits (Type I)
            print("\n*********************************** APPLYING DD TO SUBCIRCUITS ***********************************")
            dd_subcircuits = apply_dd_to_subcircuits(subcircuits, dd_sequence_type="XY4", logger=logger)
            
            # 3. Apply DD to the full circuit first, then cut it (Type II)
            print("\n*********************************** APPLYING DD TO FULL CIRCUIT THEN CUTTING ***********************************")
            # Reuse the optimization settings and device constraints from pre_processing
            optimization_settings = OptimizationParameters(seed=111)
            device_constraints = DeviceConstraints(qubits_per_subcircuit=2)
            dd_first_subcircuits, dd_first_subobservables, dd_first_partitioned_problem, metadata_dd_first = \
                pre_process_dd_first(original_circuit, observable, optimization_settings, device_constraints, "XY4", logger)
            
            # 4. Generate the cutting experiments (regular subcircuits - no DD)
            print("\n*********************************** GENERATING CUTTING EXPERIMENTS WITHOUT DD ***********************************")
            subexperiments_no_dd, coefficients_no_dd = generate_cutting_experiments(
                circuits=subcircuits, 
                observables=subobservables, 
                num_samples=10
            )
            
            # 5. Generate the cutting experiments (subcircuits with DD - DD after cutting)
            print("\n*********************************** GENERATING CUTTING EXPERIMENTS WITH DD (APPLIED AFTER CUTTING) ***********************************")
            subexperiments_with_dd, coefficients_with_dd = generate_cutting_experiments(
                circuits=dd_subcircuits, 
                observables=subobservables, 
                num_samples=10
            )
            
            # 6. Generate the cutting experiments (DD first, then cut - DD before cutting)
            print("\n*********************************** GENERATING CUTTING EXPERIMENTS WITH DD (APPLIED BEFORE CUTTING) ***********************************")
            subexperiments_dd_first, coefficients_dd_first = generate_cutting_experiments(
                circuits=dd_first_subcircuits, 
                observables=dd_first_subobservables, 
                num_samples=10
            )
            
            # Count total number of subexperiments for each approach
            subexperiment_count_no_dd = 0
            for i in range(len(subexperiments_no_dd)):
                subexperiment_count_no_dd += len(subexperiments_no_dd[i])
            print(f"Total subexperiments to run on backend (without DD): {subexperiment_count_no_dd}")
            
            subexperiment_count_with_dd = 0
            for i in range(len(subexperiments_with_dd)):
                subexperiment_count_with_dd += len(subexperiments_with_dd[i])
            print(f"Total subexperiments to run on backend (with DD after cut): {subexperiment_count_with_dd}")
            
            subexperiment_count_dd_first = 0
            for i in range(len(subexperiments_dd_first)):
                subexperiment_count_dd_first += len(subexperiments_dd_first[i])
            print(f"Total subexperiments to run on backend (with DD before cut): {subexperiment_count_dd_first}")
            
            # Generate pass manager
            print("\n*********************************** TRANSPILING CIRCUITS ***********************************")
            pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
            
            # Transpile original circuit
            full_circuit_transpiled = pass_manager.run(original_circuit)
            
            # 7. Run the full circuit with noise (no cutting, no DD) for baseline comparison
            print("\n*********************************** RUNNING FULL CIRCUIT WITH NOISE (NO DD) ***********************************")
            full_circuit_task = pcs.submit_task(
                run_noisy_circuit, 
                observable, 
                backend_options, 
                full_circuit_transpiled, 
                noise_model,
                resources={'num_cpus': 1, 'num_gpus': 2, 'memory': None}
            )
            
            # NEW: Run the full circuit without noise (ideal reference)
            print("\n*********************************** RUNNING FULL CIRCUIT WITHOUT NOISE (NO DD) ***********************************")
            full_circuit_ideal_task = pcs.submit_task(
                run_noiseless_circuit, 
                observable, 
                backend_options, 
                full_circuit_transpiled, 
                resources={'num_cpus': 1, 'num_gpus': 2, 'memory': None}
            )
            
            # 8. Process and run cut circuits without DD
            print("\n*********************************** PROCESSING CUT CIRCUITS WITHOUT DD ***********************************")
            print("Transpiling cut subcircuits...")
            isa_subexperiments_no_dd = {
                label: pass_manager.run(partition_subexpts)
                for label, partition_subexpts in subexperiments_no_dd.items()
            }
            
            # 9. Process and run cut circuits with DD (applied after cutting)
            print("\n*********************************** PROCESSING CUT CIRCUITS WITH DD (APPLIED AFTER CUTTING) ***********************************")
            print("Transpiling cut subcircuits with DD...")
            isa_subexperiments_with_dd = {
                label: pass_manager.run(partition_subexpts)
                for label, partition_subexpts in subexperiments_with_dd.items()
            }
            
            # 10. Process and run cut circuits with DD applied before cutting
            print("\n*********************************** PROCESSING CUT CIRCUITS WITH DD (APPLIED BEFORE CUTTING) ***********************************")
            print("Transpiling cut subcircuits (DD first)...")
            isa_subexperiments_dd_first = {
                label: pass_manager.run(partition_subexpts)
                for label, partition_subexpts in subexperiments_dd_first.items()
            }
            
            # Execute subcircuits for all approaches WITH NOISE
            # First without DD
            print("\n*********************************** EXECUTING CUT SUBCIRCUITS WITHOUT DD ***********************************")
            tasks_no_dd = []
            with Batch(backend=backend) as batch:
                sampler = SamplerV2(mode=batch)
                for label, subsystem_subexpts in isa_subexperiments_no_dd.items():
                    for ss in subsystem_subexpts:
                        task_future = pcs.submit_task(
                            execute_sampler, 
                            sampler, 
                            label, 
                            [ss], 
                            shots=2**14, 
                            resources={'num_cpus': 1, 'num_gpus': 1, 'memory': None}
                        )
                        tasks_no_dd.append(task_future)
            
            # Then with DD applied after cutting
            print("\n*********************************** EXECUTING CUT SUBCIRCUITS WITH DD (APPLIED AFTER CUTTING) ***********************************")
            tasks_with_dd = []
            with Batch(backend=backend) as batch:
                sampler = SamplerV2(mode=batch)
                for label, subsystem_subexpts in isa_subexperiments_with_dd.items():
                    for ss in subsystem_subexpts:
                        task_future = pcs.submit_task(
                            execute_sampler, 
                            sampler, 
                            label, 
                            [ss], 
                            shots=2**14, 
                            resources={'num_cpus': 1, 'num_gpus': 1, 'memory': None}
                        )
                        tasks_with_dd.append(task_future)
            
            # Then with DD applied before cutting
            print("\n*********************************** EXECUTING CUT SUBCIRCUITS WITH DD (APPLIED BEFORE CUTTING) ***********************************")
            tasks_dd_first = []
            with Batch(backend=backend) as batch:
                sampler = SamplerV2(mode=batch)
                for label, subsystem_subexpts in isa_subexperiments_dd_first.items():
                    for ss in subsystem_subexpts:
                        task_future = pcs.submit_task(
                            execute_sampler, 
                            sampler, 
                            label, 
                            [ss], 
                            shots=2**14, 
                            resources={'num_cpus': 1, 'num_gpus': 1, 'memory': None}
                        )
                        tasks_dd_first.append(task_future)
            
            # NEW: Execute subcircuits WITHOUT NOISE (ideal case)
            # First without DD
            print("\n*********************************** EXECUTING CUT SUBCIRCUITS WITHOUT NOISE (NO DD) ***********************************")
            tasks_no_dd_ideal = []
            with Batch(backend=AerSimulator(**backend_options["backend_options"])) as batch:
                sampler = SamplerV2(mode=batch)
                for label, subsystem_subexpts in isa_subexperiments_no_dd.items():
                    for ss in subsystem_subexpts:
                        task_future = pcs.submit_task(
                            execute_sampler, 
                            sampler, 
                            label, 
                            [ss], 
                            shots=2**14, 
                            resources={'num_cpus': 1, 'num_gpus': 1, 'memory': None}
                        )
                        tasks_no_dd_ideal.append(task_future)
            
            # Then with DD applied after cutting (without noise)
            print("\n*********************************** EXECUTING CUT SUBCIRCUITS WITHOUT NOISE (WITH DD AFTER CUTTING) ***********************************")
            tasks_with_dd_ideal = []
            with Batch(backend=AerSimulator(**backend_options["backend_options"])) as batch:
                sampler = SamplerV2(mode=batch)
                for label, subsystem_subexpts in isa_subexperiments_with_dd.items():
                    for ss in subsystem_subexpts:
                        task_future = pcs.submit_task(
                            execute_sampler, 
                            sampler, 
                            label, 
                            [ss], 
                            shots=2**14, 
                            resources={'num_cpus': 1, 'num_gpus': 1, 'memory': None}
                        )
                        tasks_with_dd_ideal.append(task_future)
            
            # Then with DD applied before cutting (without noise)
            print("\n*********************************** EXECUTING CUT SUBCIRCUITS WITHOUT NOISE (WITH DD BEFORE CUTTING) ***********************************")
            tasks_dd_first_ideal = []
            with Batch(backend=AerSimulator(**backend_options["backend_options"])) as batch:
                sampler = SamplerV2(mode=batch)
                for label, subsystem_subexpts in isa_subexperiments_dd_first.items():
                    for ss in subsystem_subexpts:
                        task_future = pcs.submit_task(
                            execute_sampler, 
                            sampler, 
                            label, 
                            [ss], 
                            shots=2**14, 
                            resources={'num_cpus': 1, 'num_gpus': 1, 'memory': None}
                        )
                        tasks_dd_first_ideal.append(task_future)
            
            # Wait for results of noisy circuits
            print("\n*********************************** WAITING FOR RESULTS ***********************************")
            results_no_dd = pcs.get_results(tasks_no_dd)
            results_with_dd = pcs.get_results(tasks_with_dd)
            results_dd_first = pcs.get_results(tasks_dd_first)
            noisy_expval = pcs.get_results([full_circuit_task])[0]
            
            # Wait for results of ideal circuits
            results_no_dd_ideal = pcs.get_results(tasks_no_dd_ideal)
            results_with_dd_ideal = pcs.get_results(tasks_with_dd_ideal)
            results_dd_first_ideal = pcs.get_results(tasks_dd_first_ideal)
            ideal_expval = pcs.get_results([full_circuit_ideal_task])[0]
            
            timing_data = []  # List to store timing info

            # Process results for noisy cases
            samplePubResults_no_dd = collections.defaultdict(list)
            for result in results_no_dd:
                if result is not None:
                    label, primitive_result, submit_t, exec_t = result
                    samplePubResults_no_dd[label].extend(primitive_result._pub_results)
                    timing_data.append({
                        "method": "cut_no_dd",
                        "label": label,
                        "submit_time": submit_t,
                        "exec_time": exec_t
                    })

            samplePubResults_with_dd = collections.defaultdict(list)
            for result in results_with_dd:
                if result is not None:
                    label, primitive_result, submit_t, exec_t = result
                    samplePubResults_with_dd[label].extend(primitive_result._pub_results)
                    timing_data.append({
                        "method": "cut_with_dd_after",
                        "label": label,
                        "submit_time": submit_t,
                        "exec_time": exec_t
                    })
                    
            samplePubResults_dd_first = collections.defaultdict(list)
            for result in results_dd_first:
                if result is not None:
                    label, primitive_result, submit_t, exec_t = result
                    samplePubResults_dd_first[label].extend(primitive_result._pub_results)
                    timing_data.append({
                        "method": "cut_with_dd_before",
                        "label": label,
                        "submit_time": submit_t,
                        "exec_time": exec_t
                    })

            # Process results for noiseless cases
            samplePubResults_no_dd_ideal = collections.defaultdict(list)
            for result in results_no_dd_ideal:
                if result is not None:
                    label, primitive_result, submit_t, exec_t = result
                    samplePubResults_no_dd_ideal[label].extend(primitive_result._pub_results)
                    timing_data.append({
                        "method": "cut_no_dd_ideal",
                        "label": label,
                        "submit_time": submit_t,
                        "exec_time": exec_t
                    })

            samplePubResults_with_dd_ideal = collections.defaultdict(list)
            for result in results_with_dd_ideal:
                if result is not None:
                    label, primitive_result, submit_t, exec_t = result
                    samplePubResults_with_dd_ideal[label].extend(primitive_result._pub_results)
                    timing_data.append({
                        "method": "cut_with_dd_after_ideal",
                        "label": label,
                        "submit_time": submit_t,
                        "exec_time": exec_t
                    })

            samplePubResults_dd_first_ideal = collections.defaultdict(list)
            for result in results_dd_first_ideal:
                if result is not None:
                    label, primitive_result, submit_t, exec_t = result
                    samplePubResults_dd_first_ideal[label].extend(primitive_result._pub_results)
                    timing_data.append({
                        "method": "cut_with_dd_before_ideal",
                        "label": label,
                        "submit_time": submit_t,
                        "exec_time": exec_t
                    })
            
            # Convert to results dictionary format for noisy cases
            results_dict_no_dd = {}
            for label, samples in samplePubResults_no_dd.items():
                results_dict_no_dd[label] = PrimitiveResult(samples)
            
            results_dict_with_dd = {}
            for label, samples in samplePubResults_with_dd.items():
                results_dict_with_dd[label] = PrimitiveResult(samples)
            
            results_dict_dd_first = {}
            for label, samples in samplePubResults_dd_first.items():
                results_dict_dd_first[label] = PrimitiveResult(samples)
            
            # Convert to results dictionary format for noiseless cases
            results_dict_no_dd_ideal = {}
            for label, samples in samplePubResults_no_dd_ideal.items():
                results_dict_no_dd_ideal[label] = PrimitiveResult(samples)

            results_dict_with_dd_ideal = {}
            for label, samples in samplePubResults_with_dd_ideal.items():
                results_dict_with_dd_ideal[label] = PrimitiveResult(samples)

            results_dict_dd_first_ideal = {}
            for label, samples in samplePubResults_dd_first_ideal.items():
                results_dict_dd_first_ideal[label] = PrimitiveResult(samples)
            
            # Reconstruct expectation values for noisy cases
            print("\n*********************************** RECONSTRUCTING EXPECTATION VALUES ***********************************")
            print("Reconstructing without DD...")
            reconstructed_expvals_no_dd = reconstruct_expectation_values(
                results_dict_no_dd,
                coefficients_no_dd,
                subobservables,
            )
            
            print("Reconstructing with DD (applied after cutting)...")
            reconstructed_expvals_with_dd = reconstruct_expectation_values(
                results_dict_with_dd,
                coefficients_with_dd,
                subobservables,
            )
            
            print("Reconstructing with DD (applied before cutting)...")
            reconstructed_expvals_dd_first = reconstruct_expectation_values(
                results_dict_dd_first,
                coefficients_dd_first,
                dd_first_subobservables,
            )
            
            # Reconstruct expectation values for noiseless cases
            print("\n*********************************** RECONSTRUCTING NOISELESS EXPECTATION VALUES ***********************************")
            print("Reconstructing without DD (noiseless)...")
            reconstructed_expvals_no_dd_ideal = reconstruct_expectation_values(
                results_dict_no_dd_ideal,
                coefficients_no_dd,
                subobservables,
            )

            print("Reconstructing with DD after cutting (noiseless)...")
            reconstructed_expvals_with_dd_ideal = reconstruct_expectation_values(
                results_dict_with_dd_ideal,
                coefficients_with_dd,
                subobservables,
            )

            print("Reconstructing with DD before cutting (noiseless)...")
            reconstructed_expvals_dd_first_ideal = reconstruct_expectation_values(
                results_dict_dd_first_ideal,
                coefficients_dd_first,
                dd_first_subobservables,
            )
            
            # Calculate final expectation values for noisy cases
            final_expval_no_dd = np.dot(reconstructed_expvals_no_dd, observable.coeffs)
            final_expval_with_dd = np.dot(reconstructed_expvals_with_dd, observable.coeffs)
            final_expval_dd_first = np.dot(reconstructed_expvals_dd_first, observable.coeffs)
            
            # Calculate final expectation values for noiseless cases
            final_expval_no_dd_ideal = np.dot(reconstructed_expvals_no_dd_ideal, observable.coeffs)
            final_expval_with_dd_ideal = np.dot(reconstructed_expvals_with_dd_ideal, observable.coeffs)
            final_expval_dd_first_ideal = np.dot(reconstructed_expvals_dd_first_ideal, observable.coeffs)
            
            # Print results
            print("\n*********************************** RESULTS SUMMARY ***********************************")
            print(f"Ideal expectation value (full circuit, no noise): {np.round(ideal_expval, 8)}")
            #print(f"Noisy expectation value (full circuit, no DD): {np.round(noisy_expval, 8)}")
            #print(f"Ideal reconstructed value (cut circuit, no noise): {np.real(np.round(final_expval_no_dd_ideal, 8))}")
            print(f"Reconstructed expectation value (cut circuit without DD): {np.real(np.round(final_expval_no_dd, 8))}")
            #print(f"Ideal reconstructed value (cut circuit, DD after cutting, no noise): {np.real(np.round(final_expval_with_dd_ideal, 8))}")
            print(f"Reconstructed expectation value (cut circuit with DD after cutting): {np.real(np.round(final_expval_with_dd, 8))}")
            #print(f"Ideal reconstructed value (cut circuit, DD before cutting, no noise): {np.real(np.round(final_expval_dd_first_ideal, 8))}")
            print(f"Reconstructed expectation value (cut circuit with DD before cutting): {np.real(np.round(final_expval_dd_first, 8))}")
            
            # Create complete results data for CSV
            results_data = [
                {
                    "method": "ideal_full_no_dd",
                    "expectation value": float(np.real(ideal_expval)),
                    "description": "Full circuit without noise (ideal reference)",
                    "num_qubits": original_circuit.num_qubits,
                    "depth": original_circuit.depth(),
                    "sampling_overhead": 1.0,  # No sampling overhead for full circuit
                    "num_cuts": 0
                },
                {
                    "method": "noisy_full_no_dd",
                    "expectation value": float(np.real(noisy_expval)),
                    "description": "Full circuit with noise only (no DD)",
                    "num_qubits": original_circuit.num_qubits,
                    "depth": original_circuit.depth(),
                    "sampling_overhead": 1.0,  # No sampling overhead for full circuit
                    "num_cuts": 0
                },
                {
                    "method": "ideal_cut_no_dd",
                    "expectation value": float(np.real(final_expval_no_dd_ideal)),
                    "description": "Cut circuit without noise (no DD)",
                    "num_qubits": original_circuit.num_qubits,
                    "depth": original_circuit.depth(),
                    "sampling_overhead": np.prod([b.overhead for b in partitioned_problem.bases]),
                    "num_cuts": len(metadata_no_dd["cuts"])
                },
                {
                    "method": "noisy_cut_no_dd",
                    "expectation value": float(np.real(final_expval_no_dd)),
                    "description": "Cut circuit with noise (no DD)",
                    "num_qubits": original_circuit.num_qubits,
                    "depth": original_circuit.depth(),
                    "sampling_overhead": np.prod([b.overhead for b in partitioned_problem.bases]),
                    "num_cuts": len(metadata_no_dd["cuts"])
                },
                {
                    "method": "ideal_cut_with_dd_after",
                    "expectation value": float(np.real(final_expval_with_dd_ideal)),
                    "description": "Cut circuit without noise and DD applied after cutting",
                    "num_qubits": original_circuit.num_qubits,
                    "depth": original_circuit.depth(),
                    "sampling_overhead": np.prod([b.overhead for b in partitioned_problem.bases]),
                    "num_cuts": len(metadata_no_dd["cuts"])
                },
                {
                    "method": "noisy_cut_with_dd_after",
                    "expectation value": float(np.real(final_expval_with_dd)),
                    "description": "Cut circuit with noise and DD applied after cutting",
                    "num_qubits": original_circuit.num_qubits,
                    "depth": original_circuit.depth(),
                    "sampling_overhead": np.prod([b.overhead for b in partitioned_problem.bases]),
                    "num_cuts": len(metadata_no_dd["cuts"])
                },
                {
                    "method": "ideal_cut_with_dd_before",
                    "expectation value": float(np.real(final_expval_dd_first_ideal)),
                    "description": "Cut circuit without noise and DD applied before cutting",
                    "num_qubits": original_circuit.num_qubits,
                    "depth": original_circuit.depth(),
                    "sampling_overhead": np.prod([b.overhead for b in dd_first_partitioned_problem.bases]),
                    "num_cuts": len(metadata_dd_first["cuts"])
                },
                {
                    "method": "noisy_cut_with_dd_before",
                    "expectation value": float(np.real(final_expval_dd_first)),
                    "description": "Cut circuit with noise and DD applied before cutting",
                    "num_qubits": original_circuit.num_qubits,
                    "depth": original_circuit.depth(),
                    "sampling_overhead": np.prod([b.overhead for b in dd_first_partitioned_problem.bases]),
                    "num_cuts": len(metadata_dd_first["cuts"])
                }
            ]
            
            # Write timing and results to CSV
            write_results_to_csv(results_data, "circuit_cutting_dd_results_with_ideal.csv")
            write_timing_to_csv(timing_data, "cutting_execution_timing_with_ideal.csv")
            
            # Expanded error analysis using the ideal full circuit as reference
            print("\n*********************************** EXPANDED ERROR ANALYSIS ***********************************")
            # Use the ideal full circuit as the true reference
            true_reference = ideal_expval

            # Calculate errors against true reference
            error_ideal_cut_no_dd = abs(final_expval_no_dd_ideal - true_reference)
            error_ideal_cut_dd_after = abs(final_expval_with_dd_ideal - true_reference)
            error_ideal_cut_dd_before = abs(final_expval_dd_first_ideal - true_reference)

            error_full_no_dd = abs(noisy_expval - true_reference)
            error_cut_no_dd = abs(final_expval_no_dd - true_reference)
            error_cut_dd_after = abs(final_expval_with_dd - true_reference)
            error_cut_dd_before = abs(final_expval_dd_first - true_reference)

            print("Errors compared to ideal full circuit:")
            #print(f"Error (ideal cut circuit, no DD): {error_ideal_cut_no_dd}")
            #print(f"Error (ideal cut circuit, DD after cutting): {error_ideal_cut_dd_after}")
            #print(f"Error (ideal cut circuit, DD before cutting): {error_ideal_cut_dd_before}")
            #print(f"Error (noisy full circuit): {error_full_no_dd}")
            print(f"Error (noisy cut circuit, no DD): {error_cut_no_dd}")
            print(f"Error (noisy cut circuit, DD after cutting): {error_cut_dd_after}")
            print(f"Error (noisy cut circuit, DD before cutting): {error_cut_dd_before}")

            # Calculate cutting errors (noise-free)
            cutting_error = error_ideal_cut_no_dd
            print(f"\nCutting error (without noise): {cutting_error}")

            # Calculate noise impact on full circuit
            noise_impact_full = error_full_no_dd
            print(f"Noise impact on full circuit: {noise_impact_full}")

            # Calculate combined noise and cutting error
            combined_error_no_dd = error_cut_no_dd
            print(f"Combined noise and cutting error (no DD): {combined_error_no_dd}")

            # Calculate DD effectiveness in noise mitigation for cut circuits
            dd_effectiveness_after = (error_cut_no_dd - error_cut_dd_after) / error_cut_no_dd * 100 if error_cut_no_dd != 0 else 0
            dd_effectiveness_before = (error_cut_no_dd - error_cut_dd_before) / error_cut_no_dd * 100 if error_cut_no_dd != 0 else 0
            print(f"\nDD effectiveness (after cutting): {dd_effectiveness_after:.2f}%")
            print(f"DD effectiveness (before cutting): {dd_effectiveness_before:.2f}%")
            
            # Calculate circuit depths and sizes for comparison
            print("\n*********************************** CIRCUIT COMPLEXITY COMPARISON ***********************************")
            print(f"Full circuit (no DD) depth: {original_circuit.depth()}, size: {len(original_circuit)}")

            avg_depth_no_dd = sum(subcircuits[i].depth() for i in subcircuits) / len(subcircuits)
            avg_size_no_dd = sum(len(subcircuits[i]) for i in subcircuits) / len(subcircuits)
            print(f"Subcircuits without DD - average depth: {avg_depth_no_dd}, average size: {avg_size_no_dd}")
            
            avg_depth_dd_after = sum(dd_subcircuits[i].depth() for i in dd_subcircuits) / len(dd_subcircuits)
            avg_size_dd_after = sum(len(dd_subcircuits[i]) for i in dd_subcircuits) / len(dd_subcircuits)
            print(f"Subcircuits with DD after cutting - average depth: {avg_depth_dd_after}, average size: {avg_size_dd_after}")
            
            avg_depth_dd_before = sum(dd_first_subcircuits[i].depth() for i in dd_first_subcircuits) / len(dd_first_subcircuits)
            avg_size_dd_before = sum(len(dd_first_subcircuits[i]) for i in dd_first_subcircuits) / len(dd_first_subcircuits)
            print(f"Subcircuits with DD before cutting - average depth: {avg_depth_dd_before}, average size: {avg_size_dd_before}")
            
            print(f"Total runtime: {time.time() - start_time} seconds")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if pcs:                
                pcs.cancel()