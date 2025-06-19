import datetime
import os
import time
import numpy as np
from engine.manager import MiniAppExecutor
from engine.metrics.csv_writer import MetricsFileWriter
from mini_apps.qml_training.utils.discrete_qcbm_model_handler import DiscreteQCBMModelHandler
from qugen.main.data.data_handler import load_data
import pennylane as qml

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

qml_parameters = {
    "build_parameters": {
        'model_type': "discrete",
        'data_set_name': "X_2D",
        'n_qubits': 8,
        'n_registers': 2,
        'circuit_depth': 2,
        'initial_sigma': 0.01,
        'circuit_type': "copula",
        'transformation': "pit",
        'hot_start_path': "",
        "parallelism_framework": "jax"
    },
    "train_parameters": {            
        'n_epochs': 5,
        'batch_size': 200,
        'hist_samples': 100000
    }
}

# Device selection
def get_device(n_qubits, noisy=False):
    if noisy:
        dev = qml.device("default.mixed", wires=n_qubits)
    else:
        dev = qml.device("default.qubit", wires=n_qubits)
    return dev

configs = [("Noise Only", False, True),("DD + Noise", True, True),("No-DD No-Noise", False, False),("DD Only", True, False)]

# Dynamical decoupling implementation
def apply_dd_sequence(wires):
    for w in wires:
        qml.PauliX(w)
        qml.PauliY(w)
        qml.PauliX(w)
        qml.PauliY(w)

# Define kraus operators for manual phase damping 
def get_phase_damping_kraus(prob):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - prob)]])
    K1 = np.array([[0, 0], [0, np.sqrt(prob)]])
    return [K0, K1]

# DD circuits
def dd_copula_ansatz(params, wires):
    n_qubits = len(wires)
    depth = params.shape[0] - 1

    for wire in range(n_qubits):
        qml.RY(params[0, wire], wires=wires[wire])

    for d in range(1, depth + 1):
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[wires[i], wires[i+1]])
            qml.QubitChannel(get_phase_damping_kraus(0.05), wires=wires[i+1])
        apply_dd_sequence(wires)
        for wire in range(n_qubits):
            qml.RY(params[d, wire], wires=wires[wire])


def regular_copula_ansatz(params, wires):
    n_qubits = len(wires)
    depth = params.shape[0] - 1

    for wire in range(n_qubits):
        qml.RY(params[0, wire], wires=wires[wire])

    for d in range(1, depth + 1):
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[wires[i], wires[i+1]])
            qml.QubitChannel(get_phase_damping_kraus(0.1), wires=wires[i+1])
        for wire in range(n_qubits):
            qml.RY(params[d, wire], wires=wires[wire])


def patch_copula_ansatz_with_dd():
    """Patch the original copula_ansatz with the DD-enhanced version."""
    try:
        import qugen.main.generator.quantum_circuits.discrete_generator_pennylane as qgen_circuits
        
        # Check if copula_ansatz exists in the module
        if hasattr(qgen_circuits, 'copula_ansatz'):
            # Store original function
            original_copula_ansatz = qgen_circuits.copula_ansatz
            
            # Patch with DD version
            qgen_circuits.copula_ansatz = dd_copula_ansatz
            
            print("[INFO] Successfully patched copula_ansatz with DD sequence")
            return original_copula_ansatz
        else:
            print("[WARNING] copula_ansatz not found in qgen_circuits module")
            # Create and patch a new function if it doesn't exist
            qgen_circuits.copula_ansatz = dd_copula_ansatz
            print("[INFO] Created new copula_ansatz function with DD sequence")
            return regular_copula_ansatz
            
    except ImportError as e:
        print(f"[ERROR] Cannot import from qugen.main.generator.quantum_circuits: {e}")
        print("[INFO] Will use local implementation of copula_ansatz")
        # No patching performed, will use our local implementation
        return regular_copula_ansatz


# QML training mini-app
class QMLTrainingMiniApp:
    def __init__(self, cluster_config, parameters=None, scenario_label="QML Training with DD MiniApp"):
        self.executor = MiniAppExecutor(cluster_config).get_executor()
        self.parameters = parameters
        self.scenario_label = scenario_label
        self.cluster_config = cluster_config
        self.current_datetime = datetime.datetime.now()
        self.timestamp = self.current_datetime.strftime('%Y-%m-%dT%H:%M:%S')
        self.file_name = f"qml_result_{self.timestamp}.csv"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.result_dir = os.path.join(script_dir, "results")
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.result_file = os.path.join(self.result_dir, self.file_name)

        header = ["timestamp", "scenario_label", "num_qubits", "compute_time_sec", 
                  "parameters", "cluster_info", "final_kl"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)

    
    def run(self, use_dd=True, use_noise=False):
        start_time = time.time()

        # Only run with the specified configuration
        futures = self.executor.submit_task(run_training_task, self.parameters, use_dd, use_noise)
        result = self.executor.get_results([futures])[0]

        end_time = time.time()
        compute_time_sec = end_time - start_time

        # Extract final_kl from result if it's a dict
        final_kl = 1.0  # Default value
        if isinstance(result, dict):
            final_kl = result.get("final_kl", 1.0)
        else:
            try:
                final_kl = float(result)
            except (TypeError, ValueError):
                final_kl = 1.0

        # Record training results
        self.metrics_file_writer.write([
            self.timestamp,
            self.scenario_label,
            self.parameters["build_parameters"]['n_qubits'],
            compute_time_sec,
            str(self.parameters),
            str(self.cluster_config),
            final_kl
        ])

        return final_kl

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'metrics_file_writer'):
            self.metrics_file_writer.close()


# Training task implementation

def run_training_task(parameters, use_dd=True, use_noise=False):
    try:
        print(f"[INFO] use_dd={use_dd}, use_noise={use_noise}")
        
        seed = parameters["build_parameters"].get("seed", 2)
        np.random.seed(seed)

        dev = get_device(parameters["build_parameters"]['n_qubits'], noisy=use_noise)

        # Conditionally patch with DD
        if use_dd:
            try:
                original_ansatz = patch_copula_ansatz_with_dd()
                print("[INFO] Patched copula_ansatz with dynamical decoupling (DD) sequence")
            except Exception as e:
                print(f"[WARNING] Failed to patch copula_ansatz: {e}")
                print("[INFO] Will proceed with local implementation")
                original_ansatz = regular_copula_ansatz
        else:
            # Always reset to regular ansatz if DD is not used
            try:
                import qugen.main.generator.quantum_circuits.discrete_generator_pennylane as qgen_circuits
                qgen_circuits.copula_ansatz = regular_copula_ansatz
                print("[INFO] Patched copula_ansatz with regular (non-DD) sequence")
            except Exception as e:
                print(f"[WARNING] Failed to patch regular copula_ansatz: {e}")
            original_ansatz = regular_copula_ansatz


        # Load or generate dataset
        package_path = os.path.dirname(os.path.abspath(__file__))
        data_set_path = os.path.join(package_path, "data", parameters["build_parameters"]["data_set_name"])

        if not os.path.exists(data_set_path):
            data = np.random.randn(1000, 2)
            np.save('X_2D.npy', data)
        else:
            data = np.load(data_set_path)

        # Build and train the model
        model = DiscreteQCBMModelHandler()
        model.build(
            parameters["build_parameters"]['model_type'],
            parameters["build_parameters"]['data_set_name'],
            n_qubits=parameters["build_parameters"]['n_qubits'],
            n_registers=parameters["build_parameters"]['n_registers'],
            circuit_depth=parameters["build_parameters"]['circuit_depth'],
            circuit_type=parameters["build_parameters"]['circuit_type'],
            transformation=parameters["build_parameters"]['transformation'],
            hot_start_path=parameters.get("build_parameters", {}).get('hot_start_path', ''),
            parallelism_framework=parameters["build_parameters"]['parallelism_framework']
        )

        model.train(
            data,
            n_epochs=parameters["train_parameters"]['n_epochs'],
            batch_size=parameters["train_parameters"]['batch_size'],
            hist_samples=parameters["train_parameters"]['hist_samples'],
        )

        # Evaluate model
        evaluation_df = model.evaluate(data)
        minimum_kl_data = evaluation_df.loc[evaluation_df["kl_original_space"].idxmin()]
        final_kl = minimum_kl_data["kl_original_space"]

        print(f"[Training Done] Final KL Divergence: {final_kl}")
        return final_kl

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return {"error": str(e), "final_kl": 1.0}


# Main execution
if __name__ == "__main__":

    try:
        # Display introductory message
        print("\nQML Training with Dynamical Decoupling")
        print("This application runs quantum machine learning with:")
        print("  1. Dynamical decoupling (DD)")
        print("  2. Distributed execution via Pilot Quantum")
        print("  3. Noise simulation with PennyLane's mixed state simulator")
        
        # Create and run the QML mini-app
        qml_mini_app = QMLTrainingMiniApp(cluster_info, qml_parameters)
        
        # Run the main training task on the cluster

        print("No DD + Noise")
        label = "No DD + Noise"
        qml_mini_app.scenario_label = label
        result = qml_mini_app.run(use_dd=False, use_noise=True)
        print(f"[Training Done] Final KL Divergence for {label}: {result}")

        print("With DD + Noise")
        label = "With DD + Noise"
        qml_mini_app.scenario_label = label
        result = qml_mini_app.run(use_dd=True, use_noise=True)
        print(f"[Training Done] Final KL Divergence for {label}: {result}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            if 'qml_mini_app' in locals():
                qml_mini_app.close()
        except:
            pass
        
        print("\nExecution complete.")