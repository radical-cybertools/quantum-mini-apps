import re
import os
import csv
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


# Device selection
def get_device(n_qubits: int, noisy: bool = False, noise_level: float = 0.05) -> qml.Device:
    if noisy:
        # Use mixed state simulator for noise
        dev = qml.device("default.mixed", wires=n_qubits)
    else:
        # Use standard qubit simulator
        dev = qml.device("default.qubit", wires=n_qubits)
    return dev

def optimize_vqe(dev, hamiltonian, wires, max_steps=100, stepsize=0.1, noise_level=0.05):
    n_qubits = len(wires)
    init_params = qml.numpy.array(np.random.uniform(0, 2*np.pi, size=(2, n_qubits*3)), requires_grad=True)

    @qml.qnode(dev)
    def circuit(params):
        vqe_ansatz(params, wires=wires)
        if dev.name == "default.mixed":
            apply_noise(wires=wires, noise_level=noise_level)
        return qml.expval(hamiltonian)
    
    print(qml.draw(circuit)(init_params))

    opt = AdamOptimizer(stepsize)
    params = init_params

    for step in range(max_steps):
        params, energy = opt.step_and_cost(circuit, params)
        print(f"Step {step}: Energy = {energy:.6f}")
    return energy, params


# Quantum chemistry circuits
def vqe_ansatz(params: np.ndarray, wires: List[int]) -> None:
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


# Distributed circuit execution tasks
def run_qchem_circuit_task(parameters: Dict, hamiltonian_data: Dict, 
                         use_noise: bool = False, noise_level: float = 0.05) -> Dict:
    try:
        print(f"[INFO] Running QChem circuit with use_noise={use_noise}, noise_level={noise_level}")
        
        n_qubits = len(hamiltonian_data["observables"][0])
        if n_qubits > 12:
            print(f"[WARNING] Skipping Hamiltonian with {n_qubits} qubits — too large for memory.")
            return {"energy": None, "qubits": n_qubits, "skipped": True}

        circuit_depth = parameters['circuit_depth']
        
        # Create device
        dev = get_device(n_qubits, noisy=use_noise, noise_level=noise_level)
        
        # Extract Hamiltonian data
        coeffs = hamiltonian_data['coeffs']
        observables = hamiltonian_data['observables']
       
        print(f"[DEBUG] Coeffs: {coeffs}")
        print(f"[DEBUG] Observables: {observables}")        

        hamiltonian = construct_hamiltonian(coeffs, observables)
        
        # Generate random parameters if none provided
        if 'circuit_params' in parameters:
            params = parameters['circuit_params']
        else:
            # Each qubit needs 3 params per layer (RY, RZ, RY)
            params = np.random.uniform(0, 2*np.pi, size=(circuit_depth, n_qubits*3))
    
        # Optimize using VQE
        start_time = time.time()
        energy, optimized_params = optimize_vqe(
            dev, hamiltonian, wires=list(range(n_qubits)),
            max_steps=parameters.get("max_steps", 100),
            stepsize=parameters.get("stepsize", 0.1),
            noise_level=noise_level if use_noise else 0.0
        )
        end_time = time.time()

        # Store optimized circuit drawing
        @qml.qnode(dev)
        def final_circuit():
            vqe_ansatz(optimized_params, wires=range(n_qubits))
            return qml.expval(hamiltonian)
        
        return {
            "energy": float(energy),
            "execution_time": end_time - start_time,
            "use_noise": use_noise,
            "noise_level": noise_level,
            "bond_length": hamiltonian_data.get("bond_length", 0.0)
        }
    
    except Exception as e:
        print(f"[ERROR] Circuit execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "energy": None}

# Quantum chemistry mini app
class QChemMiniApp:
    def __init__(self, cluster_config: Dict, dataset_path: str, 
                 parameters: Optional[Dict] = None, 
                 scenario_label: str = "QChem VQE Demo"):
        
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.dataset_loader = QChemDatasetLoader(dataset_path)
        self.dataset, self.metadata = self.dataset_loader.load()
        
        # Debug dataset structure
        self.dataset_loader.debug_dataset_structure()
        
        # Set default parameters if none provided
        if parameters is None:
            n_qubits = self.dataset_loader.get_number_of_qubits()
            parameters = {
                'n_qubits': n_qubits,
                'circuit_depth': 2,
                # Each qubit needs 3 params per layer (RY, RZ, RY)
                'circuit_params': np.random.uniform(0, 2*np.pi, size=(2, n_qubits*3))
            }
            
        self.parameters = parameters
        self.scenario_label = scenario_label
        self.cluster_config = cluster_config
        
        # Set up results directory and file
        self.current_datetime = datetime.datetime.now()
        self.timestamp = self.current_datetime.strftime('%Y-%m-%dT%H:%M:%S')
        self.file_name = f"qchem_vqe_results_{self.timestamp}.csv"
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.result_dir = os.path.join(script_dir, "results")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.result_file = os.path.join(self.result_dir, self.file_name)
        
        # Create metrics file writer
        header = ["timestamp", "scenario_label", "num_qubits", "compute_time_sec", 
                  "use_noise", "noise_level", "bond_length", "energy"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)
    
    def run_experiment(self, data_index: int, use_noise: bool = False, noise_level: float = 0.05) -> Dict:
        start_time = time.time()
        
        # Get Hamiltonian data
        coeffs, observables = self.dataset_loader.get_hamiltonian(data_index)
        bond_length = self.dataset_loader.get_bond_length(data_index)
        
        hamiltonian_data = {
            'coeffs': coeffs,
            'observables': observables,
            'bond_length': bond_length
        }
        
        # Submit task to executor
        future = self.executor.submit_task(
            run_qchem_circuit_task, 
            self.parameters, 
            hamiltonian_data,
            use_noise=use_noise,
            noise_level=noise_level
        )
        
        # Get result
        result_dict = self.executor.get_results([future])[0]
        end_time = time.time()
        compute_time_sec = end_time - start_time
        
        # Extract result from dictionary
        energy_value = result_dict.get("energy", None)
        
        # Record results
        self.metrics_file_writer.write([
            self.timestamp,
            self.scenario_label,
            self.parameters['n_qubits'],
            compute_time_sec,
            use_noise,
            noise_level,
            bond_length,
            energy_value
        ])
        
        return result_dict
    
    def run_parallel_bond_length_study(self, use_noise: bool = False, 
                                      noise_level: float = 0.05) -> List[Dict]:
        start_time = time.time()
    
        # Get total number of data points
        num_data_points = self.dataset_loader.get_num_data_points()
        print(f"[INFO] Running parallel calculations for {num_data_points} bond lengths")
    
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
        
            # Submit task to executor
            futures.append(self.executor.submit_task(
                run_qchem_circuit_task, 
                self.parameters, 
                hamiltonian_data,
                use_noise=use_noise,
                noise_level=noise_level
            ))
    
        # Get all results
        results = self.executor.get_results(futures)
        end_time = time.time()
        total_compute_time = end_time - start_time
    
        # Record results
        for i, result_dict in enumerate(results):
            bond_length = self.dataset_loader.get_bond_length(i)
            energy_value = result_dict.get("energy", None)
        
            self.metrics_file_writer.write([
                self.timestamp,
                f"{self.scenario_label}_parallel",
                self.parameters['n_qubits'],
                total_compute_time / len(results),  # Approximate per-task time
                use_noise,
                noise_level,
                bond_length,
                energy_value
            ])
    
        print(f"[INFO] Completed {len(results)} parallel bond length calculations in {total_compute_time:.2f} seconds")
        return results
    
    def plot_potential_energy_surface(self, results: List[Dict], save_path: Optional[str] = None) -> None:
        # Extract bond lengths and energies
        bond_lengths = []
        energies = []
        
        for result in results:
            if "energy" in result and result["energy"] is not None:
                bond_lengths.append(result["bond_length"])
                energies.append(result["energy"])
        
        # Sort by bond length
        sorted_data = sorted(zip(bond_lengths, energies))
        bond_lengths = [x[0] for x in sorted_data]
        energies = [x[1] for x in sorted_data]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(bond_lengths, energies, 'o-', color='blue', markersize=8)
        plt.xlabel('Bond Length (Å)')
        plt.ylabel('Energy (Hartree)')
        plt.title('H₂ Potential Energy Surface from VQE')
        plt.grid(True)
        
        # Find minimum energy point
        min_energy_idx = energies.index(min(energies))
        min_bond_length = bond_lengths[min_energy_idx]
        min_energy = energies[min_energy_idx]
        
        plt.plot(min_bond_length, min_energy, 'ro', markersize=10)
        plt.annotate(f'Equilibrium: {min_bond_length:.2f} Å, {min_energy:.6f} Ha', 
                    xy=(min_bond_length, min_energy),
                    xytext=(min_bond_length+0.2, min_energy+0.01),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved potential energy surface plot to {save_path}")
        
        plt.show()
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'metrics_file_writer'):
            self.metrics_file_writer.close()
            
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
        print("\nQuantum Chemistry with Variational Quantum Eigensolver")
        print("This application demonstrates quantum chemistry calculations with:")
        print("  1. H2 molecule data from PennyLane Datasets")
        print("  2. Distributed execution via Pilot Quantum")
        print("  3. Noise simulation with PennyLane's mixed state simulator")
        print("  4. VQE optimization using Adam optimizer")
        
        # Create QChem mini-app
        qchem_app = QChemMiniApp(cluster_info, dataset_path)
        
        # Use a fixed noise level of 0.05
        NOISE_LEVEL = 0.05
        
        # Run first for a single bond length (index 0)
        #print("\nRunning VQE for single bond length...")
        #ideal_result = qchem_app.run_experiment(0, use_noise=False)
        #print(f"   Energy: {ideal_result['energy']:.6f} Hartree")
        #print(f"   Bond length: {ideal_result['bond_length']} Å")
        
        # Run parallel execution for all bond lengths
        print("\nRunning parallel VQE calculations for all bond lengths...")
        pes_results = qchem_app.run_parallel_bond_length_study(use_noise=False)
        
        # Plot the potential energy surface
        plot_path = os.path.join(qchem_app.result_dir, f"h2_pes_plot_{qchem_app.timestamp}.png")
        qchem_app.plot_potential_energy_surface(pes_results, save_path=plot_path)
        
        # Run noisy simulations
        #print("\nRunning VQE with noise simulation...")
        #noisy_result = qchem_app.run_experiment(0, use_noise=True, noise_level=NOISE_LEVEL)
        #print(f"   Energy with noise: {noisy_result['energy']:.6f} Hartree")
        #print(f"   Bond length: {noisy_result['bond_length']} Å")
        
        # Run noisy parallel execution
        #print("\nRunning parallel noisy VQE calculations...")
        #noisy_pes_results = qchem_app.run_parallel_bond_length_study(use_noise=True, noise_level=NOISE_LEVEL)
        
        # Plot the noisy potential energy surface
        #print("\nPlotting noisy potential energy surface...")
        #noisy_plot_path = os.path.join(qchem_app.result_dir, f"h2_noisy_pes_plot_{qchem_app.timestamp}.png")
        #qchem_app.plot_potential_energy_surface(noisy_pes_results, save_path=noisy_plot_path)
        
        print("\nQChem MiniApp execution completed successfully!")
        
        # Clean up resources
        qchem_app.close()
        
    except Exception as e:
        print(f"[ERROR] QChem MiniApp execution failed: {e}")
        import traceback
        traceback.print_exc()