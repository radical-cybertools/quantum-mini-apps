# QDreamer: Resource-Aware Quantum Circuit Cutting

**Version:** 0.2.0 | **Status:** 🧪 Preliminary POC | **Released:** 2026-01

> **Note:** This is a preliminary proof-of-concept implementation. The API and speedup model parameters are subject to change based on further calibration and testing.

Intelligent optimization framework that automatically detects hardware resources and optimizes quantum circuit partitioning for efficient parallel execution.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Capabilities](#core-capabilities)
4. [Architecture](#architecture)
5. [Speedup Estimation](#speedup-estimation)
6. [API Reference](#api-reference)
7. [Examples](#examples)
8. [Integration](#integration)
9. [Testing & Validation](#testing--validation)
10. [Troubleshooting](#troubleshooting)

---

## Overview

QDreamer is a resource-aware optimization framework for quantum circuit cutting that:

- **Automatically detects** available hardware resources (GPUs, CPUs, memory)
- **Analyzes** quantum circuit characteristics (qubits, depth, gates)
- **Optimizes** circuit partitioning to maximize parallel execution speedup
- **Predicts** performance using calibratable power-law models
- **Integrates** seamlessly with Qiskit and circuit cutting workflows

### Key Features

✅ **Simple circuit cutting optimization** - Find optimal subcircuit size automatically  
✅ **Hardware-aware** - Detects and adapts to available GPUs/CPUs  
✅ **Calibratable** - Fit speedup models to your experimental data  

---

## Quick Start

### Installation

```bash
cd /path/to/quantum-mini-apps
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

### Basic Usage

```python
from qiskit.circuit.library import EfficientSU2
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import QDreamerCircuitCutting
from engine.manager import MiniAppExecutor

# Create a quantum circuit
circuit = EfficientSU2(30, entanglement='linear', reps=2).decompose()
circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)

# Setup executor
cluster_config = {
    "executor": "pilot",
    "config": {
        "resource": "ssh://localhost",
        "working_directory": "/path/to/work",
        "type": "ray",
        "number_of_nodes": 1,
        "cores_per_node": 64,
        "gpus_per_node": 4,
    }
}

executor = MiniAppExecutor(cluster_config).get_executor()

# Initialize QDreamer
qdreamer = QDreamerCircuitCutting(executor=executor, circuit=circuit)

# Optimize for current hardware (one line!)
allocation = qdreamer.optimize()

print(f"Optimal subcircuit size: {allocation.subcircuit_size} qubits")
print(f"Number of cuts: {allocation.num_cuts}")
print(f"Expected speedup: {allocation.speedup_factor:.2f}x")
print(f"Parallel tasks: {allocation.num_parallel_tasks}")
```

### Integration with CircuitCuttingBuilder

```python
from mini_apps.quantum_simulation.circuit_cutting.motif import CircuitCuttingBuilder

# Get optimization result
allocation = qdreamer.optimize()

# Apply to builder
cc = (CircuitCuttingBuilder()
    .set_subcircuit_size(allocation.subcircuit_size)
    .set_base_qubits(circuit.num_qubits)
    .set_observables(["Z" + "I" * (circuit.num_qubits - 1)])
    .set_num_samples(10000)
    .build(executor))

# Run optimized circuit cutting
with cc:
    cc.run()
```

---

## Core Capabilities

### 🔍 Automatic Resource Detection

QDreamer automatically discovers and profiles your hardware resources:

**GPU Detection:**
- Detects NVIDIA GPUs via `nvidia-smi`
- Identifies GPU model names and memory capacity
- Automatically falls back to CPU if no GPU available

**CPU & Memory:**
- Identifies physical and logical CPU cores
- Tracks total and available system memory
- Multi-node cluster support

```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import ResourceDetector

detector = ResourceDetector()
profile = detector.get_local_resources()
print(profile)
```

**Example Output:**
```
ResourceProfile:
  GPUs: 4 x NVIDIA A100-SXM4-40GB (40960 MB)
  CPUs: 64 physical / 128 logical
  Memory: 220.0 GB available / 251.2 GB total
```

### 📊 Circuit Analysis

Extracts comprehensive circuit characteristics to inform optimization:

```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import CircuitAnalyzer

analyzer = CircuitAnalyzer()
chars = analyzer.analyze_circuit(circuit)
print(chars)
```

**Example Output:**
```
CircuitCharacteristics:
  Qubits: 30
  Depth: 37
  Gates: 238 total (180 1Q, 58 2Q, 58 CNOT)
  2Q Gate Ratio: 24.37%
  Entanglement Density: 1.93
```

### ⚡ Intelligent Optimization

QDreamer evaluates different subcircuit sizes and selects the configuration that **maximizes speedup** relative to running the full circuit once.

**Memory Requirements:**

| Qubits | Memory (GB) | Device      |
|--------|-------------|-------------|
| 20     | 0.016       | CPU/GPU     |
| 24     | 0.268       | CPU/GPU     |
| 28     | 4.295       | GPU         |
| 30     | 17.18       | GPU (large) |
| 32     | 68.72       | GPU (80GB)  |

**Optimization Process:**
1. Detect available hardware resources
2. Analyze circuit characteristics
3. Evaluate candidate subcircuit sizes
4. Estimate speedup for each configuration
5. Return the optimal allocation

### 🎯 Clean, Consistent API

```python
# Find optimal configuration automatically
allocation = qdreamer.optimize()

# Evaluate a specific subcircuit size
allocation = qdreamer.evaluate_subcircuit_size(18)

# Analyze circuit characteristics
characteristics = qdreamer.analyze()

# Get hardware resource profile
profile = qdreamer.resource_profile

# Get all evaluated configurations
configs = qdreamer.get_all_configurations()

# Calibrate speedup estimator
params = qdreamer.calibrate_estimator(measurements)
```

### 🔧 Calibratable Speedup Models

QDreamer uses pluggable speedup estimators that can be calibrated to your specific hardware:

```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import PowerLawEstimator

# Create estimator
estimator = PowerLawEstimator()

# Calibrate from experimental measurements
measurements = [
    {"n": 36, "n_sub": 20, "k": 2, "W": 8, "S": 520.0},
    {"n": 36, "n_sub": 17, "k": 4, "W": 8, "S": 12.0},
]

params = estimator.calibrate_from_measurements(measurements)
print(f"Fitted: eta_max={params['eta_max']:.6f}, p={params['p']:.4f}")

# Use calibrated estimator
qdreamer = QDreamerCircuitCutting(executor, circuit, estimator=estimator)
allocation = qdreamer.optimize()
```

---

## Architecture

### Modular Structure

```
qdreamer/
├── qdreamer.py           - Main orchestrator
├── resource_optimizer.py - Optimization algorithms
├── data_models.py        - All data classes
├── estimators.py         - Speedup estimation plugins
├── __init__.py           - Module exports
├── example.py            - Usage examples
└── README.md             - This file
```

### Core Components

**Main Classes:**

1. **`QDreamerCircuitCutting`**: Main orchestrator that coordinates resource detection, circuit analysis, and optimization
2. **`ResourceOptimizer`**: Implements optimization algorithms to find the best subcircuit configuration
3. **`ResourceDetector`**: Detects and profiles hardware resources (GPUs, CPUs, memory)
4. **`CircuitAnalyzer`**: Extracts circuit characteristics (qubits, depth, gates, entanglement)
5. **`PowerLawEstimator`**: Predicts speedup using calibratable power-law efficiency model

**Data Classes:**

1. **`ResourceProfile`**: Represents available hardware resources
2. **`CircuitCharacteristics`**: Represents analyzed circuit properties
3. **`OptimizedAllocation`**: Final optimization recommendation with speedup prediction
4. **`CutConfiguration`**: Internal candidate configuration during optimization search
5. **`EstimatorInput`**: Input parameters for speedup estimation
6. **`SpeedupResult`**: Output from speedup estimation

---

## Speedup Estimation

### Power-Law Model

QDreamer uses a power-law efficiency decay model for speedup prediction:

```
η(R) = η_max / R^p
S = 2^(n - n_sub) × η(R) / R
```

Where:
- `η_max`: Peak parallel efficiency (default: 0.0008)
- `p`: Decay exponent (default: 0.3233)
- `R`: Number of execution rounds
- `n`: Total qubits in original circuit
- `n_sub`: Subcircuit qubits

This model captures the trade-off between:
- **Parallelism gains** from cutting (more subcircuits = more parallel tasks)
- **Sampling overhead** from circuit cutting (exponential in number of cuts)
- **Parallel efficiency** degradation with more execution rounds

### Using the Default Estimator

```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import (
    PowerLawEstimator, EstimatorInput
)

# Default estimator (uses fitted defaults)
estimator = PowerLawEstimator()

# Estimate speedup for a configuration
config = EstimatorInput(
    total_qubits=36,
    subcircuit_qubits=20,
    num_cuts=2,
    num_workers=8,
)
result = estimator.estimate_speedup(config)

print(f"Predicted speedup: {result.speedup_factor:.2f}x")
print(f"Efficiency: {result.efficiency:.4f}")
print(f"Execution rounds: {result.num_rounds}")
```

### Using Custom Parameters

```python
# Custom estimator with different parameters
estimator = PowerLawEstimator(eta_max=0.001, decay_exponent=0.4)

# Use with QDreamer
qdreamer = QDreamerCircuitCutting(
    executor=executor, 
    circuit=circuit, 
    estimator=estimator
)
```

### Calibrating from Measurements

If you have experimental data, you can calibrate the estimator to your specific hardware:

```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import PowerLawEstimator

# Create estimator
estimator = PowerLawEstimator()

# Calibrate from experimental measurements
# Each measurement needs: n (qubits), n_sub (subcircuit), k (cuts), W (workers), S (speedup)
measurements = [
    {"n": 36, "n_sub": 20, "k": 2, "W": 8, "S": 520.0},
    {"n": 36, "n_sub": 17, "k": 4, "W": 8, "S": 12.0},
    {"n": 36, "n_sub": 15, "k": 5, "W": 8, "S": 3.5},
    {"n": 34, "n_sub": 18, "k": 2, "W": 4, "S": 180.0},
]

# Fit the model to your data
params = estimator.calibrate_from_measurements(measurements)
print(f"Fitted: eta_max={params['eta_max']:.6f}, p={params['p']:.4f}")

# Now use calibrated estimator with QDreamer
qdreamer = QDreamerCircuitCutting(
    executor=executor,
    circuit=circuit,
    estimator=estimator  # Uses calibrated parameters
)
allocation = qdreamer.optimize()
```

### Calibration via QDreamer

You can also calibrate through the QDreamer interface:

```python
qdreamer = QDreamerCircuitCutting(executor, circuit)

# Calibrate the internal estimator
params = qdreamer.calibrate_estimator([
    {"n": 36, "n_sub": 20, "k": 2, "W": 8, "S": 520.0},
    {"n": 36, "n_sub": 17, "k": 4, "W": 8, "S": 12.0},
])

# Subsequent optimizations use calibrated model
allocation = qdreamer.optimize()
```

---

## API Reference

### `QDreamerCircuitCutting`

Main class for resource-aware circuit cutting optimization.

**Constructor:**
```python
QDreamerCircuitCutting(
    executor,
    circuit: Optional[QuantumCircuit] = None,
    num_samples: int = 10,
    seed: int = 111,
    use_gpu: Optional[bool] = None,
    estimator: Optional[SpeedupEstimator] = None,
)
```

**Parameters:**
- `executor`: Execution engine (Ray, Dask, etc.)
- `circuit`: Quantum circuit to optimize (optional, can be set later)
- `num_samples`: Number of samples for circuit cutting
- `seed`: Random seed for reproducibility
- `use_gpu`: Force GPU/CPU mode (None = auto-detect)
- `estimator`: Custom speedup estimator (None = use default)

**Methods:**

- **`optimize(circuit=None) -> OptimizedAllocation`**
  
  Find optimal cutting configuration for the circuit. Evaluates multiple subcircuit sizes and returns the one with maximum predicted speedup.

- **`evaluate_subcircuit_size(size, circuit=None) -> OptimizedAllocation`**
  
  Evaluate a specific subcircuit size. Useful for calibration experiments or when you want to test a specific configuration.

- **`analyze(circuit=None) -> CircuitCharacteristics`**
  
  Analyze circuit characteristics (qubits, depth, gates, entanglement).

- **`get_all_configurations() -> List[CutConfiguration]`**
  
  Get all configurations evaluated in the last optimization run.

- **`calibrate_estimator(measurements) -> Dict`**
  
  Calibrate the speedup estimator from experimental data. Returns fitted parameters.

**Properties:**

- **`resource_profile -> ResourceProfile`**: Hardware resources (cached after first access)

### Data Classes

**`OptimizedAllocation`**

Represents the optimized resource allocation recommendation.

```python
@dataclass
class OptimizedAllocation:
    subcircuit_size: int           # Qubits per subcircuit
    num_cuts: int                  # Number of cuts required
    num_parallel_tasks: int        # Total parallel tasks to execute
    use_gpu: bool = False          # GPU vs CPU mode
    sampling_overhead: float = 1.0 # Sampling overhead factor
    metadata: Dict = {}            # Additional metadata
    speedup_factor: float = 1.0    # Speedup vs full circuit (>1 is faster)

    @property
    def is_beneficial(self) -> bool:
        """True if speedup > 1.0"""
```

**`CutConfiguration`**

Internal representation of a candidate configuration during optimization.

```python
@dataclass
class CutConfiguration:
    subcircuit_size: int     # Qubits per subcircuit
    num_cuts: int            # Number of cuts
    num_tasks: int           # Total parallel tasks
    sampling_overhead: float # Sampling overhead factor
    speedup_factor: float    # Predicted speedup
    metadata: Dict           # Additional metadata
```

**`EstimatorInput`**

Input parameters for speedup estimation.

```python
@dataclass
class EstimatorInput:
    total_qubits: int           # Original circuit qubits
    subcircuit_qubits: int      # Subcircuit qubits
    num_cuts: int               # Number of cuts
    num_workers: int            # Parallel workers available
    num_tasks: Optional[int]    # Actual tasks (optional, computed if None)
```

**`SpeedupResult`**

Output from speedup estimation.

```python
@dataclass
class SpeedupResult:
    speedup_factor: float    # Predicted speedup
    efficiency: float = 0.0  # Parallel efficiency
    num_rounds: int = 1      # Execution rounds required
```

### Available Imports

```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import (
    # Main API
    QDreamerCircuitCutting,
    ResourceOptimizer,
    predict_cutting,

    # Component classes
    ResourceDetector,
    CircuitAnalyzer,

    # Data models
    ResourceProfile,
    CircuitCharacteristics,
    OptimizedAllocation,
    CutConfiguration,
    EstimatorInput,
    SpeedupResult,

    # Speedup Estimators
    SpeedupEstimator,
    PowerLawEstimator,
    EstimatorRegistry,
    fit_efficiency_power_law,
)
```

---

## Examples

### Example 1: Basic Optimization

```python
from qiskit.circuit.library import EfficientSU2
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import QDreamerCircuitCutting

# Create circuit
circuit = EfficientSU2(28, entanglement='linear', reps=2).decompose()
circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)

# Optimize
qdreamer = QDreamerCircuitCutting(executor, circuit)
allocation = qdreamer.optimize()

print(f"Subcircuit size: {allocation.subcircuit_size}q")
print(f"Cuts: {allocation.num_cuts}")
print(f"Speedup: {allocation.speedup_factor:.2f}x")
```

### Example 2: Evaluate Specific Subcircuit Sizes

```python
# Test specific configurations for calibration
qdreamer = QDreamerCircuitCutting(executor, circuit)

for size in [20, 18, 15, 12]:
    allocation = qdreamer.evaluate_subcircuit_size(size)
    print(f"{size}q: {allocation.num_cuts} cuts, {allocation.speedup_factor:.2f}x speedup")
```

### Example 3: Custom Calibrated Estimator

```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import (
    QDreamerCircuitCutting, PowerLawEstimator
)

# Create and calibrate estimator
estimator = PowerLawEstimator()
estimator.calibrate_from_measurements([
    {"n": 36, "n_sub": 20, "k": 2, "W": 8, "S": 520.0},
    {"n": 36, "n_sub": 17, "k": 4, "W": 8, "S": 12.0},
])

# Use calibrated estimator
qdreamer = QDreamerCircuitCutting(executor, circuit, estimator=estimator)
allocation = qdreamer.optimize()
```

### Example 4: Compare Different Circuit Sizes

```python
circuit_sizes = [20, 24, 28, 32]

for size in circuit_sizes:
    circuit = EfficientSU2(size, entanglement='linear', reps=2).decompose()
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)

    qdreamer = QDreamerCircuitCutting(executor, circuit)
    allocation = qdreamer.optimize()

    print(f"{size}q: {allocation.subcircuit_size}q subcircuits, "
          f"{allocation.num_cuts} cuts, {allocation.speedup_factor:.2f}x speedup")
```

### Example 5: Full Integration

```python
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import QDreamerCircuitCutting
from mini_apps.quantum_simulation.circuit_cutting.motif import CircuitCuttingBuilder
import datetime

# Create circuit
circuit = EfficientSU2(26, entanglement='linear', reps=2).decompose()
circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)

# Get QDreamer recommendations
qdreamer = QDreamerCircuitCutting(executor, circuit)
allocation = qdreamer.optimize()

print(f"Recommended subcircuit size: {allocation.subcircuit_size}")
print(f"Expected cuts: {allocation.num_cuts}")
print(f"Parallel tasks: {allocation.num_parallel_tasks}")
print(f"Expected speedup: {allocation.speedup_factor:.2f}x")

# Build and run circuit cutting
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cc = (CircuitCuttingBuilder()
    .set_subcircuit_size(allocation.subcircuit_size)
    .set_base_qubits(circuit.num_qubits)
    .set_observables(["Z" + "I" * (circuit.num_qubits - 1)])
    .set_num_samples(10000)
    .set_result_file(f"qdreamer_result_{timestamp}.csv")
    .build(executor))

with cc:
    cc.run()
```

---

## Integration

### With MiniAppExecutor

```python
cluster_config = {
    "executor": "pilot",
    "config": {
        "resource": "ssh://localhost",
        "type": "ray",
        "number_of_nodes": 1,
        "cores_per_node": 64,
        "gpus_per_node": 4,
    }
}
executor = MiniAppExecutor(cluster_config).get_executor()
qdreamer = QDreamerCircuitCutting(executor, circuit)
```

### With CircuitCuttingBuilder

```python
allocation = qdreamer.optimize()
cc = (CircuitCuttingBuilder()
    .set_subcircuit_size(allocation.subcircuit_size)
    .set_base_qubits(circuit.num_qubits)
    .build(executor))
```

---

## Testing & Validation

### Quick Tests

```bash
# Set environment
export PYTHONPATH=/path/to/quantum-mini-apps/src:$PYTHONPATH

# Test imports
python3 -c "from mini_apps.quantum_simulation.circuit_cutting.qdreamer import QDreamerCircuitCutting; print('✓ Import successful')"

# Test OptimizedAllocation
python3 -c "
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import OptimizedAllocation
alloc = OptimizedAllocation(
    subcircuit_size=18, num_cuts=3,
    num_parallel_tasks=120, use_gpu=True, speedup_factor=2.5
)
print(f'✓ Is beneficial: {alloc.is_beneficial}')
print(f'✓ Speedup: {alloc.speedup_factor:.1f}x')
"

# Test PowerLawEstimator
python3 -c "
from mini_apps.quantum_simulation.circuit_cutting.qdreamer import PowerLawEstimator, EstimatorInput
estimator = PowerLawEstimator()
result = estimator.estimate_speedup(EstimatorInput(
    total_qubits=36, subcircuit_qubits=20, num_cuts=2, num_workers=8
))
print(f'✓ Speedup prediction: {result.speedup_factor:.2f}x')
"
```

### Run Examples

```bash
python -m mini_apps.quantum_simulation.circuit_cutting.qdreamer.example
```

---

## Troubleshooting

### Issue: "No NVIDIA GPUs detected"

**Solution**: Ensure nvidia-smi is installed. The module automatically falls back to CPU.

```bash
nvidia-smi  # Test if nvidia-smi works
```

### Issue: ModuleNotFoundError

**Solution**: Set PYTHONPATH:

```bash
export PYTHONPATH=/path/to/quantum-mini-apps/src:$PYTHONPATH
```

### Issue: Poor speedup predictions

**Solution**: Calibrate the estimator with your own measurements:

```python
estimator = PowerLawEstimator()
estimator.calibrate_from_measurements(your_measurements)
qdreamer = QDreamerCircuitCutting(executor, circuit, estimator=estimator)
```

---

## Hardware Compatibility

### Tested Platforms

- ✅ **NERSC Perlmutter**: NVIDIA A100 40GB/80GB
- ✅ **Local workstations**: NVIDIA GPUs with CUDA support
- ✅ **CPU-only systems**: Automatic fallback to CPU execution

### Requirements

**Software:**
- Python 3.8+
- Qiskit 1.0+
- qiskit-addon-cutting
- psutil, numpy

**Hardware:**
- **GPU (optional)**: NVIDIA GPU with CUDA support
- **CPU**: Multi-core processor
- **Memory**: Sufficient RAM based on circuit size

---

## Contributing

Contributions are welcome! Please ensure:
1. Code follows existing style conventions
2. All functions include docstrings
3. Examples are updated accordingly

---

## Citation

```bibtex
@software{qdreamer_circuit_cutting,
  title={QDreamer: Resource-Aware Quantum Circuit Cutting Optimization},
  author={Quantum Mini-Apps Team},
  year={2025},
  version={0.2.0},
  url={https://github.com/your-repo/quantum-mini-apps}
}
```

---

**QDreamer v0.2 - Preliminary POC for calibratable quantum circuit cutting optimization** 🧪
