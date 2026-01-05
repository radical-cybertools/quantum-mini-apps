"""
Data Models for QDreamer Circuit Cutting Optimizer

This module defines all data classes used throughout QDreamer:
- ResourceProfile: Hardware resource information
- CircuitCharacteristics: Quantum circuit metrics
- OptimizedAllocation: Optimization results and recommendations
- CutConfiguration: Result of analyzing a specific subcircuit size
- EstimatorInput: Input configuration for speedup estimation
- SpeedupResult: Result of speedup estimation

These dataclasses provide clean interfaces between QDreamer components
and make it easy to serialize/deserialize optimization results.

Author: QDreamer Team
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============================================================================
# Hardware Resource Models
# ============================================================================

@dataclass
class ResourceProfile:
    """
    Represents the hardware resources available on the local machine or cluster.

    Attributes:
        num_gpus: Number of available GPUs (per node)
        gpu_memory_mb: List of GPU memory in MB for each GPU
        gpu_names: List of GPU model names
        num_cpu_cores_physical: Number of physical CPU cores (per node)
        num_cpu_cores_logical: Number of logical CPU cores with hyperthreading (per node)
        total_memory_gb: Total system memory in GB (per node)
        available_memory_gb: Currently available system memory in GB (per node)
        number_of_nodes: Number of nodes in cluster (default 1 for single node)
        gpus_per_node: Explicit GPUs per node (derived from num_gpus if not set)
        cpus_per_node: Explicit CPUs per node (derived from num_cpu_cores_physical if not set)
    """
    num_gpus: int = 0
    gpu_memory_mb: List[int] = field(default_factory=list)
    gpu_names: List[str] = field(default_factory=list)
    num_cpu_cores_physical: int = 0
    num_cpu_cores_logical: int = 0
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    number_of_nodes: int = 1
    gpus_per_node: int = 0
    cpus_per_node: int = 0

    def __post_init__(self):
        """Set derived fields if not explicitly provided."""
        if self.gpus_per_node == 0:
            self.gpus_per_node = self.num_gpus
        if self.cpus_per_node == 0:
            self.cpus_per_node = self.num_cpu_cores_physical

    @property
    def total_gpus(self) -> int:
        """Total GPUs across all nodes."""
        return self.gpus_per_node * self.number_of_nodes

    @property
    def total_cpus(self) -> int:
        """Total physical CPU cores across all nodes."""
        return self.cpus_per_node * self.number_of_nodes

    @property
    def has_gpu(self) -> bool:
        """Whether GPU resources are available."""
        return self.num_gpus > 0

    def __str__(self):
        gpu_info = f"{self.num_gpus} x {self.gpu_names[0]} ({self.gpu_memory_mb[0]} MB)" if self.num_gpus > 0 else "No GPU"
        if self.number_of_nodes > 1:
            return (f"ResourceProfile(\n"
                    f"  Nodes: {self.number_of_nodes}\n"
                    f"  GPUs: {gpu_info} per node (Total: {self.total_gpus})\n"
                    f"  CPUs: {self.num_cpu_cores_physical} physical / {self.num_cpu_cores_logical} logical per node (Total: {self.total_cpus})\n"
                    f"  Memory: {self.available_memory_gb:.1f} GB available / {self.total_memory_gb:.1f} GB total per node\n"
                    f")")
        else:
            return (f"ResourceProfile(\n"
                    f"  GPUs: {gpu_info}\n"
                    f"  CPUs: {self.num_cpu_cores_physical} physical / {self.num_cpu_cores_logical} logical\n"
                    f"  Memory: {self.available_memory_gb:.1f} GB available / {self.total_memory_gb:.1f} GB total\n"
                    f")")


# ============================================================================
# Circuit Analysis Models
# ============================================================================

@dataclass
class CircuitCharacteristics:
    """
    Represents the characteristics of a quantum circuit.

    Attributes:
        num_qubits: Number of qubits in the circuit
        depth: Circuit depth (number of time steps)
        total_gates: Total number of gates
        two_qubit_gates: Number of two-qubit gates
        cnot_gates: Number of CNOT/CX gates specifically
        single_qubit_gates: Number of single-qubit gates
        circuit: Optional reference to the actual QuantumCircuit object
    """
    num_qubits: int
    depth: int
    total_gates: int
    two_qubit_gates: int
    cnot_gates: int
    single_qubit_gates: int
    circuit: Optional[Any] = None  # QuantumCircuit object (Any to avoid import)

    @property
    def gate_ratio_2q(self) -> float:
        """Ratio of two-qubit gates to total gates"""
        return self.two_qubit_gates / self.total_gates if self.total_gates > 0 else 0.0

    @property
    def entanglement_density(self) -> float:
        """Measure of circuit entanglement based on 2-qubit gates per qubit"""
        return self.two_qubit_gates / self.num_qubits if self.num_qubits > 0 else 0.0

    def __str__(self):
        return (f"CircuitCharacteristics(\n"
                f"  Qubits: {self.num_qubits}\n"
                f"  Depth: {self.depth}\n"
                f"  Gates: {self.total_gates} total ({self.single_qubit_gates} 1Q, {self.two_qubit_gates} 2Q, {self.cnot_gates} CNOT)\n"
                f"  2Q Gate Ratio: {self.gate_ratio_2q:.2%}\n"
                f"  Entanglement Density: {self.entanglement_density:.2f}\n"
                f")")


# ============================================================================
# Optimization Result Models
# ============================================================================

@dataclass
class OptimizedAllocation:
    """
    Represents the optimized resource allocation for circuit cutting.

    Attributes:
        subcircuit_size: Recommended number of qubits per subcircuit
        num_cuts: Number of cuts required
        num_parallel_tasks: Total number of parallel tasks to execute
        use_gpu: Whether to use GPU acceleration
        sampling_overhead: Sampling overhead factor from circuit cutting
        metadata: Additional optimization metadata
        speedup_factor: Predicted speedup vs full circuit execution (>1 is faster)
    """
    subcircuit_size: int
    num_cuts: int
    num_parallel_tasks: int
    use_gpu: bool = False
    sampling_overhead: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    speedup_factor: float = 1.0

    @property
    def is_beneficial(self) -> bool:
        """Whether this allocation provides a speedup (speedup > 1.0)."""
        return self.speedup_factor > 1.0

    def __str__(self):
        device = "GPU" if self.use_gpu else "CPU"
        return (f"OptimizedAllocation(\n"
                f"  Subcircuit Size: {self.subcircuit_size} qubits\n"
                f"  Cuts: {self.num_cuts}\n"
                f"  Parallel Tasks: {self.num_parallel_tasks} ({device})\n"
                f"  Sampling Overhead: {self.sampling_overhead:.2f}x\n"
                f"  Speedup Factor: {self.speedup_factor:.2f}x\n"
                f")")


# ============================================================================
# Cut Configuration Models
# ============================================================================

@dataclass
class CutConfiguration:
    """
    Result of analyzing a specific subcircuit size during optimization.
    
    Attributes:
        subcircuit_size: Number of qubits per subcircuit
        num_cuts: Number of cuts required
        num_tasks: Total number of parallel tasks to execute
        sampling_overhead: Sampling overhead factor from circuit cutting
        speedup_factor: Predicted speedup vs full circuit execution
        metadata: Additional optimization metadata
    """
    subcircuit_size: int
    num_cuts: int
    num_tasks: int
    sampling_overhead: float
    speedup_factor: float
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# Speedup Estimator Models
# ============================================================================

@dataclass
class EstimatorInput:
    """
    Input configuration for speedup estimation.
    
    Attributes:
        total_qubits: Number of qubits in the original full circuit (n)
        subcircuit_qubits: Number of qubits in each subcircuit after cutting (n_sub)
        num_cuts: Number of wire cuts (k). Each cut creates 9x sampling overhead
        num_workers: Number of parallel workers (GPUs or CPU cores)
        num_tasks: If known, the actual number of tasks from generate_cutting_experiments.
                   If None, will be estimated as 9^num_cuts
    """
    total_qubits: int
    subcircuit_qubits: int
    num_cuts: int
    num_workers: int
    num_tasks: Optional[int] = None


@dataclass
class SpeedupResult:
    """
    Result of speedup estimation.
    
    Attributes:
        speedup_factor: Predicted speedup ratio (values > 1 indicate cutting is faster)
        efficiency: Parallel efficiency η at the given configuration
        num_rounds: Number of execution rounds R = ceil(num_tasks / num_workers)
    """
    speedup_factor: float
    efficiency: float = 0.0
    num_rounds: int = 1
