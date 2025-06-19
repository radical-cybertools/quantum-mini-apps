import re
import os
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from typing import Dict, List, Tuple, Optional, Any
from pennylane.optimize import AdamOptimizer
from engine.manager import MiniAppExecutor
from engine.metrics.csv_writer import MetricsFileWriter
from scipy.stats import entropy

# Quantum chemistry dataset handling
class QChemDatasetLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_file = os.path.join(dataset_path, "dataset.json")
        self.meta_file = os.path.join(dataset_path, "meta.json")
        self.dataset = None
        self.metadata = None
        self.dataset_keys = None
        self.hamiltonians = []  # Store processed Hamiltonians
        self.bond_lengths = []  # Store bond lengths
        
    def load(self) -> Tuple[Dict, Dict]:
        try:
            with open(self.dataset_file, 'r') as f:
                self.dataset = json.load(f)
            
            with open(self.meta_file, 'r') as f:
                self.metadata = json.load(f)
                
            print(f"[INFO] Successfully loaded dataset from {self.dataset_path}")
            print(f"[INFO] Dataset contains {len(self.dataset)} data points")
            print(f"[INFO] Molecule: {self.metadata.get('name', 'Unknown')}")
            print(f"[DEBUG] Dataset type: {type(self.dataset)}")
            
            # Process the dataset based on its structure
            self._process_dataset()
            
            return self.dataset, self.metadata
        
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            raise
        
    def _process_dataset(self):
        try:
            if isinstance(self.dataset, dict) and 'data' in self.dataset:
                for entry in self.dataset["data"]:
                    try:
                        bond_length = float(entry["parameters"]["bondlength"])
                        term_str = entry["extra"]["hamiltonianTerms"]
                        # TEMPORARILY assume 2 qubits, just for parsing
                        coeffs, observables = self.parse_hamiltonian_terms(term_str, n_qubits=2)
                        n_qubits = len(observables[0]) # Use real size after parsing
                        self.hamiltonians.append({
                            "coeffs": coeffs,
                            "observables": observables
                        })
                        self.bond_lengths.append(bond_length)
                    except Exception as e:
                        print(f"[WARNING] Failed to parse real Hamiltonian data. Using dummy data. Reason: {e}")
                        self.hamiltonians.append({
                            "coeffs": [1.0, -0.5, 0.3],
                            "observables": [["Z", "I"], ["I", "Z"], ["Z", "Z"]]
                        })
                        self.bond_lengths.append(0.5)
            if not self.hamiltonians:
                print("[WARNING] Could not extract any Hamiltonian data. Creating dummy data for testing.")
                for i in range(5):
                    self.hamiltonians.append({
                        'coeffs': [1.0, -0.5, 0.3],
                        'observables': [['Z', 'I'], ['I', 'Z'], ['Z', 'Z']]
                    })
                    self.bond_lengths.append(0.5 + i * 0.2)
            print(f"[INFO] Processed {len(self.hamiltonians)} data points with corresponding bond lengths.")
        except Exception as e:
            print(f"[ERROR] Error processing dataset: {e}")
            import traceback
            traceback.print_exc()
            print("[WARNING] Creating dummy data as fallback.")
            self.hamiltonians = []
            self.bond_lengths = []
            for i in range(5):
                self.hamiltonians.append({
                    'coeffs': [1.0, -0.5, 0.3],
                    'observables': [['Z', 'I'], ['I', 'Z'], ['Z', 'Z']]
                })
                self.bond_lengths.append(0.5 + i * 0.2)
    
    def parse_hamiltonian_terms(self, term_str: str, n_qubits: int = 8):
        coeffs = []
        observables = []
        for line in term_str.strip().split("\n"):
            match = re.match(r"\(([^)]+)\)\s+\[([^\]]+)\]", line.strip())
            if match:
                coeff = float(match.group(1))
                terms = match.group(2).split()

                # Dynamically determine required qubit count
                max_index = max([int(p[1:]) for p in terms]) if terms else 0
                vec_len = max(n_qubits, max_index + 1)
                pauli_vec = ['I'] * vec_len

                for pauli in terms:
                    p_type = pauli[0]
                    p_index = int(pauli[1:])
                    pauli_vec[p_index] = p_type

                coeffs.append(coeff)
                observables.append(pauli_vec)
        return coeffs, observables
    
    def get_hamiltonian(self, index: int) -> Tuple[List[float], List[List[str]]]:
        if self.dataset is None:
            self.load()
        
        if index >= len(self.hamiltonians):
            raise IndexError(f"Index {index} out of range for dataset with {len(self.hamiltonians)} entries")
        
        hamiltonian_data = self.hamiltonians[index]
        return hamiltonian_data['coeffs'], hamiltonian_data['observables']
    
    def get_bond_length(self, index: int) -> float:
        if self.dataset is None:
            self.load()
        
        if index >= len(self.bond_lengths):
            raise IndexError(f"Index {index} out of range for dataset with {len(self.bond_lengths)} entries")
            
        return self.bond_lengths[index]
    
    def get_number_of_qubits(self) -> int:
        if self.metadata is None:
            self.load()
            
        num_qubits = self.metadata.get("num_qubits", None)
        
        # If not specified in metadata, try to infer from the Hamiltonians
        if num_qubits is None and self.hamiltonians:
            # Get first Hamiltonian's observables
            _, observables = self.get_hamiltonian(0)
            if observables and isinstance(observables[0], list):
                # Number of qubits is the length of the first observable term
                num_qubits = len(observables[0])
                print(f"[INFO] Inferred {num_qubits} qubits from Hamiltonian observables")
            else:
                # Default to 2 qubits for H2 molecule
                num_qubits = 2
                print(f"[WARNING] Could not infer number of qubits. Defaulting to {num_qubits}")
        
        return num_qubits or 2  # Default to 2 qubits for H2 molecule
    
    def get_num_data_points(self) -> int:
        if self.dataset is None:
            self.load()
            
        return len(self.hamiltonians)

    def debug_dataset_structure(self):
        if self.dataset is None:
            self.load()
            
        print(f"[DEBUG] Dataset type: {type(self.dataset)}")
        
        try:
            if isinstance(self.dataset, dict):
                keys = list(self.dataset.keys())
                print(f"[DEBUG] Dataset has {len(keys)} keys")
                print(f"[DEBUG] First few keys: {keys[:3]}")
                
                # Print structure of first data point
                if keys:
                    first_key = keys[0]
                    print(f"[DEBUG] Structure of first data point (key={first_key}):")
                    value = self.dataset[first_key]
                    print(f"[DEBUG]   Type: {type(value)}")
                    
                    # Handle different value types
                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"[DEBUG]   {k}: {type(v)}")
                            if isinstance(v, list) and len(v) > 0:
                                print(f"[DEBUG]     First element type: {type(v[0])}")
                                if len(v) > 1:
                                    print(f"[DEBUG]     List length: {len(v)}")
                    elif isinstance(value, list):
                        print(f"[DEBUG]   List length: {len(value)}")
                        if value:
                            print(f"[DEBUG]   First element type: {type(value[0])}")
                    elif isinstance(value, str):
                        print(f"[DEBUG]   String value: {value[:50]}{'...' if len(value) > 50 else ''}")
                    else:
                        print(f"[DEBUG]   Value: {value}")
            else:
                print(f"[DEBUG] Dataset is not a dictionary. Type: {type(self.dataset)}")
                
            # Print processed data summary
            print(f"[DEBUG] Processed {len(self.hamiltonians)} Hamiltonians")
            print(f"[DEBUG] Processed {len(self.bond_lengths)} bond lengths")
            
            if self.hamiltonians and self.bond_lengths:
                print(f"[DEBUG] First Hamiltonian coeffs: {self.hamiltonians[0]['coeffs'][:3]}...")
                print(f"[DEBUG] First bond length: {self.bond_lengths[0]}")
                
        except Exception as e:
            print(f"[DEBUG] Error in debug_dataset_structure: {e}")


def get_device(n_qubits: int, noisy: bool = False, noise_level: float = 0.05) -> qml.Device:
    if noisy:
        # Use mixed state simulator for noise
        dev = qml.device("default.mixed", wires=n_qubits, shots=1000)
    else:
        # Use standard qubit simulator with shots for counts
        dev = qml.device("default.qubit", wires=n_qubits, shots=1000)
    return dev

# Dynamical decoupling implementation
def apply_dd_sequence(wires) -> None:
    for w in wires:
        qml.PauliX(w)
        qml.PauliY(w)
        qml.PauliX(w)
        qml.PauliY(w)

def optimize_vqe(dev, hamiltonian, wires, max_steps=25, stepsize=0.1, use_dd=False, noise_level=0.05):
    n_qubits = len(wires)
    init_params = qml.numpy.array(np.random.uniform(0, 2*np.pi, size=(2, n_qubits*3)), requires_grad=True)

    # Define cost function for VQE
    @qml.qnode(dev)
    def cost_function(params):
        vqe_ansatz(params, wires=range(n_qubits), use_dd=use_dd)
        
        if dev.name == "default.mixed":
            apply_noise(wires=list(range(n_qubits)), noise_level=noise_level)
            
        return qml.expval(hamiltonian)
    
    # Setup optimizer
    opt = AdamOptimizer(stepsize=stepsize)
    params = init_params
    
    # Store optimization history
    energies = []
    
    # Run optimization
    for step in range(max_steps):
        params = opt.step(cost_function, params)
        energy = cost_function(params)
        energies.append(energy)
        
        if step % 5 == 0:
            print(f"Step {step}: Energy = {energy:.6f}")
    
    # Return final energy and optimized parameters
    final_energy = cost_function(params)
    return final_energy, params

def jensen_shannon_divergence(p, q):
    # Ensure distributions sum to 1
    p = np.array(p)
    q = np.array(q)

    if np.sum(p) != 0:
        p = p / np.sum(p)
    if np.sum(q) != 0:
        q = q / np.sum(q)

    # Calculate midpoint distribution
    m = 0.5 * (p + q)

    # Calculate JS divergence using KL divergence
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))

    # Ensure the result is a finite number
    if np.isnan(jsd) or np.isinf(jsd):
        return 0.0
    
    return jsd

# Quantum chemistry circuits
def vqe_ansatz(params: np.ndarray, wires: List[int], use_dd: bool = False, dd_inserted: bool = False) -> None:
    n_qubits = len(wires)
    n_layers = params.shape[0]
    
    # Apply parameterized rotations and entangling layers
    for layer in range(n_layers):
        # Rotation layer
        param_width = params.shape[1]
        for i in range(n_qubits):
            if i*3 + 2 >= param_width:
                break  # prevent out-of-bounds access
            qml.RY(params[layer, i*3], wires=wires[i])
            qml.RZ(params[layer, i*3+1], wires=wires[i])
            qml.RY(params[layer, i*3+2], wires=wires[i])
        
        # Entangling layer
        for i in range(n_qubits):
            qml.CNOT(wires=[wires[i], wires[(i+1) % n_qubits]])
    
    # Apply DD sequence if enabled and not already inserted
        if use_dd and not dd_inserted:
            #print(f"[DEBUG] Inserting DD sequence on wires {wires}")
            apply_dd_sequence(wires)

# Noise model
def get_phase_damping_kraus(prob):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - prob)]])
    K1 = np.array([[0, 0], [0, np.sqrt(prob)]])
    return [K0, K1]

def apply_noise(wires: List[int], noise_level: float = 0.05) -> None:
    for wire in wires:
        qml.QubitChannel(get_phase_damping_kraus(noise_level), wires=wire)

# Hamiltonian construction
def construct_hamiltonian(coeffs: List[float], observables: List[List[str]]) -> qml.Hamiltonian:
    obs_list = []

    for obs_terms in observables:
        pauli_product = []
        for wire, term in enumerate(obs_terms):
            if term == "I":
                continue
            elif term == "X":
                pauli_product.append(qml.PauliX(wire))
            elif term == "Y":
                pauli_product.append(qml.PauliY(wire))
            elif term == "Z":
                pauli_product.append(qml.PauliZ(wire))
            else:
                raise ValueError(f"Unknown Pauli term: {term}")
        
        if pauli_product:
            obs = qml.prod(*pauli_product) if len(pauli_product) > 1 else pauli_product[0]
        else:
            obs = qml.Identity(0)  # Default to Identity on wire 0 if all were "I"
        
        obs_list.append(obs)

    if len(coeffs) != len(obs_list):
        raise ValueError(f"Mismatch: {len(coeffs)} coeffs vs {len(obs_list)} observables")

    return qml.Hamiltonian(coeffs, obs_list)

# Modified function to run all three scenarios in a single task
def run_combined_qchem_circuit_task(parameters: Dict, hamiltonian_data: Dict, 
                                   noise_level: float = 0.05) -> Dict:
    try:
        print(f"[INFO] Running combined QChem circuit with noise_level={noise_level}")
        
        n_qubits = len(hamiltonian_data["observables"][0])
        if n_qubits > 12:
            print(f"[WARNING] Skipping Hamiltonian with {n_qubits} qubits — too large for memory.")
            return {
                "ideal": {"energy": None, "skipped": True},
                "noisy": {"energy": None, "skipped": True},
                "noisy_dd": {"energy": None, "skipped": True},
                "qubits": n_qubits
            }

        circuit_depth = parameters['circuit_depth']
        
        # Create devices for all three scenarios
        dev_ideal = get_device(n_qubits, noisy=False)
        dev_noisy = get_device(n_qubits, noisy=True, noise_level=noise_level)
        dev_noisy_dd = get_device(n_qubits, noisy=True, noise_level=noise_level)
        
        # Extract Hamiltonian data
        coeffs = hamiltonian_data['coeffs']
        observables = hamiltonian_data['observables']
        hamiltonian = construct_hamiltonian(coeffs, observables)
        
        # Generate random parameters if none provided
        if 'circuit_params' in parameters and parameters['circuit_params'] is not None:
            params = parameters['circuit_params']
        else:
            # Each qubit needs 3 params per layer (RY, RZ, RY)
            params = np.random.uniform(0, 2*np.pi, size=(circuit_depth, n_qubits*3))
    
        # Run all three scenarios with timing
        results = {}
        
        # 1. Ideal scenario
        start_time = time.time()
        ideal_energy, ideal_params = optimize_vqe(
            dev_ideal, hamiltonian, wires=list(range(n_qubits)),
            max_steps=parameters.get("max_steps", 25),
            stepsize=parameters.get("stepsize", 0.1),
            use_dd=False,
            noise_level=0.0
        )
        ideal_time = time.time() - start_time
        results["ideal"] = {
            "energy": float(ideal_energy),
            "execution_time": ideal_time,
            "params": ideal_params
        }
        
        # 2. Noisy scenario
        start_time = time.time()
        noisy_energy, noisy_params = optimize_vqe(
            dev_noisy, hamiltonian, wires=list(range(n_qubits)),
            max_steps=parameters.get("max_steps", 25),
            stepsize=parameters.get("stepsize", 0.1),
            use_dd=False,
            noise_level=noise_level
        )
        noisy_time = time.time() - start_time
        results["noisy"] = {
            "energy": float(noisy_energy),
            "execution_time": noisy_time,
            "params": noisy_params
        }
        
        # 3. Noisy with DD scenario
        start_time = time.time()
        noisy_dd_energy, noisy_dd_params = optimize_vqe(
            dev_noisy_dd, hamiltonian, wires=list(range(n_qubits)),
            max_steps=parameters.get("max_steps", 25),
            stepsize=parameters.get("stepsize", 0.1),
            use_dd=True,
            noise_level=noise_level
        )
        noisy_dd_time = time.time() - start_time
        results["noisy_dd"] = {
            "energy": float(noisy_dd_energy),
            "execution_time": noisy_dd_time,
            "params": noisy_dd_params
        }

        # Calculate JSD metrics using the best parameters from ideal scenario
        try:
            # Create circuits for state measurements
            @qml.qnode(dev_ideal)
            def circuit_ideal_probs(params):
                vqe_ansatz(params, wires=range(n_qubits), use_dd=False)
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            @qml.qnode(dev_noisy)
            def circuit_noisy_probs(params):
                vqe_ansatz(params, wires=range(n_qubits), use_dd=False)
                if dev_noisy.name == "default.mixed":
                    apply_noise(wires=range(n_qubits), noise_level=noise_level)
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            @qml.qnode(dev_noisy_dd)
            def circuit_dd_probs(params):
                vqe_ansatz(params, wires=range(n_qubits), use_dd=True)
                if dev_noisy_dd.name == "default.mixed":
                    apply_noise(wires=range(n_qubits), noise_level=noise_level)
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            # Get expectation values using the ideal parameters
            expvals_ideal = circuit_ideal_probs(ideal_params)
            expvals_noisy = circuit_noisy_probs(ideal_params)
            expvals_dd = circuit_dd_probs(ideal_params)
            
            # Convert expectation values to approximate state probabilities
            def expvals_to_probs(expvals):
                # Convert single-qubit Z expectations to probabilities of states
                single_probs = [[(1 + ez)/2, (1 - ez)/2] for ez in expvals]
                
                # Simplified approach: use product state approximation
                n_states = 2**len(expvals)
                all_probs = np.zeros(n_states)
                
                for state_idx in range(n_states):
                    # Convert index to bit string
                    bitstring = format(state_idx, f'0{len(expvals)}b')
                    
                    # Calculate probability of this bitstring
                    prob = 1.0
                    for q_idx, bit in enumerate(bitstring):
                        bit_val = int(bit)
                        prob *= single_probs[q_idx][bit_val]
                    
                    all_probs[state_idx] = prob
                
                return all_probs
            
            # Get approximate state probabilities
            probs_ideal = expvals_to_probs(expvals_ideal)
            probs_noisy = expvals_to_probs(expvals_noisy)
            probs_dd = expvals_to_probs(expvals_dd)
            
            # Calculate JSD
            jsd_noisy_vs_ideal = jensen_shannon_divergence(probs_ideal, probs_noisy)
            jsd_dd_vs_ideal = jensen_shannon_divergence(probs_ideal, probs_dd)
            jsd_improvement = max(0, (jsd_noisy_vs_ideal - jsd_dd_vs_ideal) / max(jsd_noisy_vs_ideal, 1e-10))
            
            # Add JSD metrics to results
            results["jsd_metrics"] = {
                "jsd_noisy_vs_ideal": jsd_noisy_vs_ideal,
                "jsd_dd_vs_ideal": jsd_dd_vs_ideal,
                "jsd_improvement": jsd_improvement
            }
            
        except Exception as e:
            print(f"[WARNING] Error calculating JSD: {e}")
            import traceback
            traceback.print_exc()
            results["jsd_metrics"] = {
                "jsd_noisy_vs_ideal": None,
                "jsd_dd_vs_ideal": None,
                "jsd_improvement": None
            }
        
        # Add common metadata
        results["metadata"] = {
            "bond_length": hamiltonian_data.get("bond_length", 0.0),
            "noise_level": noise_level,
            "n_qubits": n_qubits
        }
        
        return results
    
    except Exception as e:
        print(f"[ERROR] Combined circuit execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "ideal": {"energy": None},
            "noisy": {"energy": None},
            "noisy_dd": {"energy": None}
        }

# Modified QChemDDMiniApp class
class OptimizedQChemDDMiniApp:
    def __init__(self, cluster_config: Dict, dataset_path: str, 
                 parameters: Optional[Dict] = None, 
                 scenario_label: str = "QChem DD Demo"):
        
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.dataset_loader = QChemDatasetLoader(dataset_path)
        self.dataset, self.metadata = self.dataset_loader.load()
        
        # Set default parameters if none provided
        if parameters is None:
            n_qubits = self.dataset_loader.get_number_of_qubits()
            parameters = {
                'n_qubits': n_qubits,
                'circuit_depth': 2,
                'max_steps': 25,  # Reduced from default for faster execution
                'stepsize': 0.1,
                # Each qubit needs 3 params per layer (RY, RZ, RY)
                'circuit_params': np.random.uniform(0, 2*np.pi, size=(2, n_qubits*3))
            }
            
        self.parameters = parameters
        self.scenario_label = scenario_label
        self.cluster_config = cluster_config
        
        # Set up results directory and file
        self.current_datetime = datetime.datetime.now()
        self.timestamp = self.current_datetime.strftime('%Y-%m-%dT%H:%M:%S')
        self.file_name = f"qchem_dd_results_{self.timestamp}.csv"
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.result_dir = os.path.join(script_dir, "results")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.result_file = os.path.join(self.result_dir, self.file_name)
        
        # Create metrics file writer
        header = ["timestamp", "scenario", "num_qubits", "compute_time_sec", 
                  "use_dd", "use_noise", "noise_level", "bond_length", "energy", 
                  "jsd_noisy_vs_ideal", "jsd_dd_vs_ideal", "jsd_improvement"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)
    
    def run_optimized_parallel_study(self, noise_level: float = 0.05) -> Dict:
        start_time = time.time()
    
        # Get total number of data points
        num_data_points = self.dataset_loader.get_num_data_points()
        print(f"[INFO] Running optimized parallel calculations for {num_data_points} bond lengths with all three scenarios")
    
        # Create list of futures for all tasks
        futures = []
        for data_index in range(num_data_points):
            # Get Hamiltonian data
            coeffs, observables = self.dataset_loader.get_hamiltonian(data_index)
            bond_length = self.dataset_loader.get_bond_length(data_index)
        
            hamiltonian_data = {
                'coeffs': coeffs,
                'observables': observables,
                'bond_length': bond_length
            }
        
            # Submit task to executor - one task per bond length
            futures.append(self.executor.submit_task(
                run_combined_qchem_circuit_task, 
                self.parameters, 
                hamiltonian_data,
                noise_level=noise_level
            ))
    
        # Get all results
        results = self.executor.get_results(futures)
        end_time = time.time()
        total_compute_time = end_time - start_time
    
        print(f"[INFO] Completed {len(results)} bond length calculations in {total_compute_time:.2f} seconds")
        
        # Organize results by bond length
        organized_results = {}
        for i, result_dict in enumerate(results):
            bond_length = self.dataset_loader.get_bond_length(i)
            organized_results[bond_length] = result_dict
            
            # Record results in CSV file
            if "error" not in result_dict:
                # Ideal scenario
                self.metrics_file_writer.write([
                    self.timestamp,
                    "ideal",
                    self.parameters['n_qubits'],
                    result_dict["ideal"].get("execution_time", 0),
                    False,  # use_dd
                    False,  # use_noise
                    0.0,    # noise_level
                    bond_length,
                    result_dict["ideal"].get("energy", None),
                    None,   # jsd metrics not applicable
                    None,
                    None
                ])
                
                # Noisy scenario
                self.metrics_file_writer.write([
                    self.timestamp,
                    "noisy",
                    self.parameters['n_qubits'],
                    result_dict["noisy"].get("execution_time", 0),
                    False,  # use_dd
                    True,   # use_noise
                    noise_level,
                    bond_length,
                    result_dict["noisy"].get("energy", None),
                    result_dict.get("jsd_metrics", {}).get("jsd_noisy_vs_ideal", None),
                    None,
                    None
                ])
                
                # Noisy with DD scenario
                self.metrics_file_writer.write([
                    self.timestamp,
                    "noisy_dd",
                    self.parameters['n_qubits'],
                    result_dict["noisy_dd"].get("execution_time", 0),
                    True,   # use_dd
                    True,   # use_noise
                    noise_level,
                    bond_length,
                    result_dict["noisy_dd"].get("energy", None),
                    None,
                    result_dict.get("jsd_metrics", {}).get("jsd_dd_vs_ideal", None),
                    result_dict.get("jsd_metrics", {}).get("jsd_improvement", None)
                ])
    
        return organized_results
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'metrics_file_writer'):
            self.metrics_file_writer.close()
    
    def calculate_and_display_dd_improvement(self, results: Dict) -> None:
        # Sort results by bond length
        bond_lengths = sorted(results.keys())
        
        # Prepare data for analysis
        ideal_energies = []
        noisy_energies = []
        dd_energies = []
        
        jsd_noisy_vs_ideal = []
        jsd_dd_vs_ideal = []
        jsd_improvements = []
        
        for bond_length in bond_lengths:
            result = results[bond_length]
            
            ideal_energy = result["ideal"].get("energy", None)
            noisy_energy = result["noisy"].get("energy", None) 
            dd_energy = result["noisy_dd"].get("energy", None)
            
            ideal_energies.append(ideal_energy)
            noisy_energies.append(noisy_energy)
            dd_energies.append(dd_energy)
            
            # Extract JSD metrics
            jsd_metrics = result.get("jsd_metrics", {})
            jsd_noisy_vs_ideal.append(jsd_metrics.get("jsd_noisy_vs_ideal", None))
            jsd_dd_vs_ideal.append(jsd_metrics.get("jsd_dd_vs_ideal", None))
            jsd_improvements.append(jsd_metrics.get("jsd_improvement", None))
        
        # Calculate error metrics
        errors_noisy = []
        errors_dd = []
        improvements = []
        
        for i in range(len(bond_lengths)):
            if ideal_energies[i] is None or noisy_energies[i] is None or dd_energies[i] is None:
                errors_noisy.append(None)
                errors_dd.append(None)
                improvements.append(None)
                continue
                
            error_noisy = abs(ideal_energies[i] - noisy_energies[i])
            error_dd = abs(ideal_energies[i] - dd_energies[i])
            
            errors_noisy.append(error_noisy)
            errors_dd.append(error_dd)
            
            # Calculate percent improvement
            if error_noisy > 0:
                improvement = (error_noisy - error_dd) / error_noisy * 100
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        # Print summary
        print("\nDynamic Decoupling Improvement Summary:")
        print("Bond Length (Å) | Energy Error w/o DD | Energy Error with DD | Energy Improvement (%) | JSD w/o DD | JSD with DD | JSD Improvement (%)")
        print("-" * 120)
        
        for i in range(len(bond_lengths)):
            # Handle potential None values with safe formatting
            bond_length_str = f"{bond_lengths[i]:.4f}" if bond_lengths[i] is not None else "N/A"
            error_noisy_str = f"{errors_noisy[i]:.6f}" if errors_noisy[i] is not None else "N/A"
            error_dd_str = f"{errors_dd[i]:.6f}" if errors_dd[i] is not None else "N/A"
            improvement_str = f"{improvements[i]:.2f}" if improvements[i] is not None else "N/A"
            
            # Handle JSD metrics which might be None
            jsd_noisy_str = f"{jsd_noisy_vs_ideal[i]:.6f}" if jsd_noisy_vs_ideal[i] is not None else "N/A"
            jsd_dd_str = f"{jsd_dd_vs_ideal[i]:.6f}" if jsd_dd_vs_ideal[i] is not None else "N/A"
            
            # Calculate JSD improvement percentage safely
            jsd_impr_pct = jsd_improvements[i] * 100 if jsd_improvements[i] is not None else None
            jsd_impr_str = f"{jsd_impr_pct:.2f}" if jsd_impr_pct is not None else "N/A"
            
            print(f"{bond_length_str}        | {error_noisy_str}    | {error_dd_str}    | {improvement_str}                | {jsd_noisy_str}   | {jsd_dd_str}  | {jsd_impr_str}")
        
        # Plot energy curves
        plt.figure(figsize=(10, 6))
        
        # Filter out None values
        valid_indices = [i for i in range(len(bond_lengths)) if ideal_energies[i] is not None]
        valid_bond_lengths = [bond_lengths[i] for i in valid_indices]
        valid_ideal = [ideal_energies[i] for i in valid_indices]
        valid_noisy = [noisy_energies[i] for i in valid_indices]
        valid_dd = [dd_energies[i] for i in valid_indices]
        
        if valid_bond_lengths:
            plt.plot(valid_bond_lengths, valid_ideal, 'b-', label='Ideal')
            plt.plot(valid_bond_lengths, valid_noisy, 'r--', label='Noisy')
            plt.plot(valid_bond_lengths, valid_dd, 'g-.', label='Noisy+DD')
            
            plt.xlabel('Bond Length (Å)')
            plt.ylabel('Energy')
            plt.title('H2 Energy vs Bond Length')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.result_dir, f'energy_curves_{self.timestamp}.png'))
        else:
            print("[WARNING] No valid energy data points to plot")

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
    
    # Path to the H2 molecule dataset
    dataset_path = "/scratch/4891333/pq/examples/DD/pennylane-datasets/content/qchem/h2-molecule"
    
    try:
        # Display introductory message
        print("\nOptimized Quantum Chemistry with Dynamic Decoupling and VQE")
        print("This application demonstrates quantum chemistry calculations with:")
        print("  1. H2 molecule data from PennyLane Datasets")
        print("  2. X-Y-X-Y dynamical decoupling (DD) sequences")
        print("  3. Efficient parallel execution of all bond lengths")
        print("  4. Combined ideal, noisy, and noisy+DD scenarios in a single run")
        
        # Create optimized QChem DD mini-app
        qchem_dd_app = OptimizedQChemDDMiniApp(cluster_info, dataset_path, 
                                              parameters={
                                                  'n_qubits': 2,  # H2 molecule typically needs 2 qubits
                                                  'circuit_depth': 2,
                                                  'max_steps': 25,  # Reduce for faster execution
                                                  'stepsize': 0.1,
                                                  'circuit_params': None  # Will be randomly generated
                                              })
        
        # Use a fixed noise level of 0.05
        NOISE_LEVEL = 0.05
        
        print("\nRunning optimized parallel bond length study...")
        # Run parallel bond length studies for all three cases in a single operation
        results = qchem_dd_app.run_optimized_parallel_study(noise_level=NOISE_LEVEL)
        
        # Analyze and display results
        qchem_dd_app.calculate_and_display_dd_improvement(results)

        print("\nExecution complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'qchem_dd_app' in locals():
                qchem_dd_app.close()
        except:
            pass