# QDreamer: Resource-Aware Quantum Circuit Cutting

**Version:** 0.4.0 

QDreamer is a resource-aware optimization framework for quantum circuit cutting that automatically detects hardware resources and optimizes circuit partitioning for parallel execution.

---

## Module Structure

```
qdreamer/
├── __init__.py                    # Main entry point (re-exports all public API)
├── README.md
├── core/                          # Shared core components
│   ├── __init__.py
│   ├── data_models.py            # ResourceProfile, OptimizedAllocation, etc.
│   └── detector.py               # ResourceDetector, CircuitAnalyzer
├── tools/                         # Optimization tools
│   ├── __init__.py
│   └── circuit_cutting_resource_optimizer/
│       ├── __init__.py
│       ├── resource_optimizer.py          # ResourceOptimizer class
│       ├── qdreamer_circuit_cutting.py    # QDreamerCircuitCutting class
│       └── estimators/                    # Speedup estimation plugins
│           ├── __init__.py
│           ├── base.py                    # SpeedupEstimator abstract base
│           └── power_law.py               # PowerLawEstimator, registry
└── examples/                      # Example scripts
    ├── __init__.py
    ├── basic.py                  # Basic analysis (no executor needed)
    ├── executor.py               # Full execution pipeline with Pilot-Quantum
    └── calibration.py            # Model calibration experiments
```

---

## Overview

QDreamer provides:

- **Resource Detection**: Automatically detects GPUs, CPUs, and memory
- **Circuit Analysis**: Extracts circuit characteristics (qubits, depth, gates)
- **Optimization**: Finds optimal subcircuit size for circuit cutting
- **Calibration**: Fits speedup models to experimental measurements

### Installation

```bash
cd /path/to/quantum-mini-apps
pip install --upgrade .
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

---

## Examples

### 1. Basic Analysis - Understanding When Cutting Helps

**File:** `examples/basic.py`

Demonstrates scenarios where circuit cutting is beneficial vs not beneficial, without requiring an executor:

```bash
python -m mini_apps.quantum_simulation.circuit_cutting.qdreamer.examples.basic
```

This example shows three scenarios:
- **36-qubit / 8 workers** → Cutting IS beneficial (large speedup)
- **20-qubit / 4 workers** → Cutting is NOT beneficial (overhead too high)
- **20-qubit / 64 workers** → Cutting becomes beneficial with more workers

**Key insight**: Circuit cutting provides speedup when:
- The circuit is large enough that `2^(n - n_sub)` factor dominates
- There are enough parallel workers to execute subcircuit tasks efficiently

**Code snippet:**
```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import (
    ResourceOptimizer,
    ResourceProfile,
)

# Define available resources
resource_profile = ResourceProfile(
    num_cpu_cores_physical=8,
    available_memory_gb=48.0,
    cpus_per_node=8,
)

# Run optimization
optimizer = ResourceOptimizer(resource_profile=resource_profile, circuit=circuit)
allocation = optimizer.find_best_configuration()

print(f"Subcircuit size: {allocation.subcircuit_size} qubits")
print(f"Predicted speedup: {allocation.speedup_factor:.2f}x")
```

---

### 2. Executor Integration - Complete Execution Pipeline

**File:** `examples/executor.py`

Demonstrates the full workflow: optimization + actual circuit cutting execution with Pilot-Quantum:

```bash
python -m mini_apps.quantum_simulation.circuit_cutting.qdreamer.examples.executor
```

This example shows:
1. Setting up a Pilot-Quantum executor
2. Using QDreamer to find optimal cutting configuration
3. Executing circuit cutting with CircuitCuttingBuilder
4. Collecting and displaying results

**Code snippet:**
```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import QDreamerCircuitCutting
from mini_apps.quantum_simulation.circuit_cutting.motif import CircuitCuttingBuilder
from engine.manager import MiniAppExecutor

# 1. Create executor
executor = MiniAppExecutor(cluster_config).get_executor()

# 2. Use QDreamer to optimize
qdreamer = QDreamerCircuitCutting(executor=executor, circuit=circuit)
allocation = qdreamer.optimize()

# 3. Execute with optimized parameters
cc = (CircuitCuttingBuilder()
    .set_subcircuit_size(allocation.subcircuit_size)
    .set_base_qubits(num_qubits)
    .set_observables(["Z" + "I" * (num_qubits - 1)])
    .set_num_samples(10000)
    .build(executor))

with cc:
    results = cc.run()
```

---

### 3. Model Calibration - Fitting the Speedup Model

**File:** `examples/calibration.py`

Full-scale calibration experiments for fitting the power-law speedup model to your hardware:

```bash
python -m mini_apps.quantum_simulation.circuit_cutting.qdreamer.examples.calibration
```

**Configuration options:**
```python
# Circuit configuration
QUBIT_RANGE = [36]              # Number of qubits to test
CIRCUIT_TYPE = "EfficientSU2"   # Circuit type
CIRCUIT_DEPTH = 1               # Circuit depth (reps)

# Calibration settings
CALIBRATION_MODE = True
NUM_REPEATS = 3                 # Repeats per configuration

# Subcircuit sizes to test (determines number of cuts)
# For 36 qubits with linear entanglement:
#   18q subcircuit = 1 cut  (2 partitions)
#   12q subcircuit = 2 cuts (3 partitions)
#   9q subcircuit  = 3 cuts (4 partitions)
CALIBRATION_SUBCIRCUIT_SIZES = [18, 12, 9, 8]

# Parallelization configurations: (nodes, cores_per_node, gpus_per_node)
PARALLELIZATION_CONFIGS = [
    (1, 224, 0),   # CPU: 224 cores
    (1, 4, 4),     # GPU: 4 GPUs
    (1, 8, 8),     # GPU: 8 GPUs
]
```

**Using calibrated estimators:**
```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import PowerLawEstimator

estimator = PowerLawEstimator()

# Calibrate from measurements
# Each measurement: n (qubits), n_sub (subcircuit), k (cuts), W (workers), S (speedup)
measurements = [
    {"n": 36, "n_sub": 18, "k": 1, "W": 8, "S": 520.0},
    {"n": 36, "n_sub": 12, "k": 2, "W": 8, "S": 248.0},
    {"n": 36, "n_sub": 9,  "k": 3, "W": 8, "S": 45.0},
]

params = estimator.calibrate_from_measurements(measurements)
print(f"Fitted: eta_max={params['eta_max']:.6f}, p={params['p']:.4f}")

# Use calibrated estimator with QDreamer
qdreamer = QDreamerCircuitCutting(executor, circuit, estimator=estimator)
allocation = qdreamer.optimize()
```

---

## API Reference

### Core Classes

| Class | Location | Description |
|-------|----------|-------------|
| `QDreamerCircuitCutting` | `tools.circuit_cutting_resource_optimizer` | Main interface with executor integration |
| `ResourceOptimizer` | `tools.circuit_cutting_resource_optimizer` | Standalone optimizer (no executor needed) |
| `ResourceDetector` | `core.detector` | Hardware resource detection |
| `CircuitAnalyzer` | `core.detector` | Circuit characteristics analysis |

### Estimators

| Class | Location | Description |
|-------|----------|-------------|
| `SpeedupEstimator` | `tools.circuit_cutting_resource_optimizer.estimators` | Abstract base class for estimators |
| `PowerLawEstimator` | `tools.circuit_cutting_resource_optimizer.estimators` | Power-law efficiency decay model |
| `EstimatorRegistry` | `tools.circuit_cutting_resource_optimizer.estimators` | Plugin registry for estimators |

### Data Classes

| Class | Location | Description |
|-------|----------|-------------|
| `ResourceProfile` | `core.data_models` | Hardware resources (GPUs, CPUs, memory) |
| `OptimizedAllocation` | `core.data_models` | Optimization results (subcircuit size, cuts, speedup) |
| `CircuitCharacteristics` | `core.data_models` | Circuit metrics (qubits, depth, gates) |
| `CutConfiguration` | `core.data_models` | Single cut configuration evaluation |
| `EstimatorInput` | `core.data_models` | Input for speedup estimation |
| `SpeedupResult` | `core.data_models` | Speedup estimation result |

### Key Methods

**QDreamerCircuitCutting:**
- `optimize(quick=True)` - Find optimal cutting configuration
- `evaluate_subcircuit_size(size)` - Evaluate specific subcircuit size
- `calibrate_estimator(measurements)` - Calibrate from experimental data

**ResourceOptimizer:**
- `find_best_configuration()` - Search for optimal configuration
- `get_all_configurations()` - Get all evaluated configurations

**PowerLawEstimator:**
- `estimate_speedup(config)` - Estimate speedup for given configuration
- `calibrate_from_measurements(measurements)` - Fit model to experimental data

---

## Speedup Model

QDreamer uses a power-law efficiency decay model:

```
efficiency(R) = η_max / R^p
speedup = 2^(n - n_sub) × efficiency(R) / R
```

Where:
- `η_max`: Peak parallel efficiency (fitted parameter, default: 0.0002)
- `p`: Decay exponent (fitted parameter, default: 0.7630)
- `R`: Number of execution rounds = ceil(num_tasks / num_workers)
- `n`: Total qubits in original circuit
- `n_sub`: Qubits per subcircuit after cutting
- `num_tasks`: Number of subcircuit tasks = 9^num_cuts

### Cut Estimation (Linear Entanglement)

For circuits with linear entanglement (like EfficientSU2):
```
num_cuts = ceil(n / n_sub) - 1
```

Examples for 36-qubit circuit:
- 18q subcircuits → 1 cut (2 partitions)
- 12q subcircuits → 2 cuts (3 partitions)
- 9q subcircuits → 3 cuts (4 partitions)

---

## Requirements

**Software:**
- Python 3.10+
- Qiskit 1.x
- qiskit-addon-cutting
- psutil, numpy

**Hardware:**
- GPU (optional): NVIDIA GPU with CUDA support
- CPU: Multi-core processor

---

## References

- Quantum Mini-Apps: [arXiv:2412.18519](https://arxiv.org/abs/2412.18519)
- Pilot-Quantum: [arXiv:2405.07333](https://arxiv.org/abs/2405.07333)
