# QML Classifier Mini-App

## Overview
The QML Classifier Mini-App is a quantum machine learning benchmarking tool designed to evaluate the performance of quantum classifiers using different optimization techniques and hardware configurations. It supports GPU acceleration, Just-In-Time (JIT) compilation, and vectorized mapping (vmap) for improved performance.

## Key Features
- GPU-accelerated quantum circuit execution
- JIT compilation support for optimized performance
- Vectorized batch processing with vmap
- Configurable circuit parameters:
  - Number of qubits
  - Circuit depth
  - Batch size
  - Number of epochs
- Comprehensive performance metrics collection
- Distributed execution capabilities using Ray

## Configuration

### Basic Configuration
Configuration can be specified either through YAML files or programmatically:

```yaml
batch_size: 80
depth: 2
device: gpu
jit: true
n_batches: 10
n_epochs: 2
n_qubits: 11
vmap: true
```

### Cluster Configuration
The mini-app supports configuration for HPC environments:

```python
cluster_info = {
    "executor": "pilot",
    "config": {
        "resource": "slurm://localhost",
        "working_directory": "/path/to/work",
        "number_of_nodes": 1,
        "cores_per_node": 256,
        "gpus_per_node": 4,
        "queue": "premium",
        "walltime": 30,
        "type": "ray"
    }
}
```

## Usage

### Command Line

```
python classifier.py
```

### Code Configuration and Execution
```python
from mini_apps.qml_classifier.classifier import QMLClassifierMiniApp

# Initialize mini-app with cluster configuration
app = QMLClassifierMiniApp(cluster_info)

# Define training configurations
configs = [{
    "n_qubits": 13,
    "depth": 1,
    "batch_size": 64,
    "n_batches": 502,
    "n_epochs": 1,
    "jit": True,
    "vmap": True,
    "device": "gpu"
}]

# Run training
app.run(configs)
```

### Performance Optimization
The mini-app supports several optimization techniques:

1. JIT Compilation:
   - Enables ahead-of-time compilation of quantum circuits
   - Reduces runtime overhead

2. Vectorized Mapping (vmap):
   - Enables parallel processing of batched inputs
   - Improves GPU utilization

3. Batch Size Tuning:
   - Configurable batch sizes for optimal performance
   - Automatic batch size adjustment based on hardware capabilities

## Implementation Details
The mini-app consists of two main components:

1. **QMLClassifierMiniApp**: Main class handling execution and metrics collection

```11:31:src/mini_apps/qml_classifier/classifier.py
class QMLClassifierMiniApp:
    def __init__(self, pilot_compute_description):
        os.makedirs(pilot_compute_description["config"]["working_directory"], exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"result_compression_{current_time}.csv"
        self.result_file = os.path.join(pilot_compute_description["config"]["working_directory"], file_name)
        print(f"Result file: {self.result_file}")
        header = ["column_1", "compute_time_sec"]
        self.metrics_file_writer = MetricsFileWriter(self.result_file, header)
        self.executor = MiniAppExecutor(pilot_compute_description).get_executor()

    def run(self, configs):
        start = perf_counter()
        futures = self.executor.submit_tasks(training, configs)
        self.executor.wait(futures)
        print("Done")
        compute_time_sec = perf_counter() - start
        self.metrics_file_writer.write([
            "worked",
            compute_time_sec])
```


2. **Configuration Management**: Flexible configuration system supporting both YAML and programmatic configuration

```1:9:src/mini_apps/qml_classifier/config.yml
batch_size: 80
depth: 2
device: gpu
jit: true
n_batches: 10
n_epochs: 2
n_qubits: 11
vmap: true

```
