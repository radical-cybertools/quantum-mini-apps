"""
QDreamer Circuit Cutting Module

This module provides intelligent resource-aware circuit cutting optimization for quantum computing.
It automatically detects local hardware resources (GPUs, CPUs, memory) and optimizes circuit partitioning
based on circuit characteristics to achieve efficient parallel execution.

This is the refactored main module that coordinates all QDreamer components:
- Resource detection and profiling
- Circuit analysis
- Resource optimization (via resource_optimizer.py)
- Pluggable speedup estimation (via estimators.py)

Author: QDreamer Team
"""

import logging
import subprocess
from typing import Dict, List, Optional

import psutil
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

# Local imports - new modular structure
from engine.base.base_motif import Motif
from .data_models import (
    ResourceProfile,
    CircuitCharacteristics,
    OptimizedAllocation,
    CutConfiguration,
)
from .resource_optimizer import ResourceOptimizer
from .estimators import SpeedupEstimator, PowerLawEstimator, EstimatorRegistry


# ============================================================================
# Resource Detection
# ============================================================================

class ResourceDetector:
    """
    Detects and profiles local hardware resources including GPUs, CPUs, and memory.
    Supports both single-node and multi-node cluster configurations.
    """

    def __init__(self, executor_config: Optional[Dict] = None):
        """
        Initialize ResourceDetector.

        Args:
            executor_config: Optional executor configuration dict with cluster info.
                           Expected to have 'config' dict with keys like:
                           - 'number_of_nodes': int
                           - 'gpus_per_node': int
                           - 'cores_per_node': int
        """
        self.logger = logging.getLogger(__name__)
        self.executor_config = executor_config

    def get_local_resources(self) -> ResourceProfile:
        """
        Detect all available local resources, considering multi-node cluster config.

        Returns:
            ResourceProfile: Complete profile of local/cluster hardware resources
        """
        profile = ResourceProfile()

        # Detect GPUs
        gpu_info = self._detect_gpus()
        profile.num_gpus = gpu_info['num_gpus']
        profile.gpu_memory_mb = gpu_info['memory_mb']
        profile.gpu_names = gpu_info['names']

        # Detect CPUs
        profile.num_cpu_cores_physical = psutil.cpu_count(logical=False) or 0
        profile.num_cpu_cores_logical = psutil.cpu_count(logical=True) or 0

        # Detect Memory
        mem = psutil.virtual_memory()
        profile.total_memory_gb = mem.total / (1024 ** 3)
        profile.available_memory_gb = mem.available / (1024 ** 3)

        # Apply executor cluster configuration if available
        if self.executor_config and 'config' in self.executor_config:
            config = self.executor_config['config']

            # Get number of nodes (default to 1 if not specified)
            profile.number_of_nodes = config.get('number_of_nodes', 1)

            # Override per-node resources if explicitly specified in config
            if 'gpus_per_node' in config:
                profile.gpus_per_node = config['gpus_per_node']
            else:
                profile.gpus_per_node = profile.num_gpus

            if 'cores_per_node' in config:
                profile.cpus_per_node = config['cores_per_node']
            else:
                profile.cpus_per_node = profile.num_cpu_cores_physical

            self.logger.info(f"Applied multi-node config: {profile.number_of_nodes} nodes")
        else:
            # No executor config - use single node with detected resources
            profile.number_of_nodes = 1
            profile.gpus_per_node = profile.num_gpus
            profile.cpus_per_node = profile.num_cpu_cores_physical

        # Warn if detected GPUs differ from configured GPUs
        if profile.num_gpus > 0 and profile.total_gpus == 0:
            self.logger.warning(
                f"GPUs detected ({profile.num_gpus}) but disabled in configuration "
                f"(gpus_per_node={profile.gpus_per_node}). GPU acceleration will not be used."
            )
        elif profile.num_gpus != profile.total_gpus and profile.num_gpus > 0:
            self.logger.info(
                f"GPU configuration: {profile.num_gpus} detected locally, "
                f"{profile.total_gpus} total across {profile.number_of_nodes} node(s)"
            )

        self.logger.info(f"Detected resources:\n{profile}")
        return profile

    def _detect_gpus(self) -> Dict[str, any]:
        """
        Detect NVIDIA GPUs using nvidia-smi.

        Returns:
            Dict with 'num_gpus', 'memory_mb', and 'names'
        """
        result = {
            'num_gpus': 0,
            'memory_mb': [],
            'names': []
        }

        try:
            # Try using nvidia-smi
            cmd = ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits']
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)

            lines = output.strip().split('\n')
            result['num_gpus'] = len(lines)

            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    result['names'].append(parts[1])
                    result['memory_mb'].append(int(parts[2]))

            self.logger.info(f"Detected {result['num_gpus']} NVIDIA GPU(s)")

        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.info("No NVIDIA GPUs detected or nvidia-smi not available")

        return result


# ============================================================================
# Circuit Analysis
# ============================================================================

class CircuitAnalyzer:
    """
    Analyzes quantum circuit characteristics relevant for cutting optimization.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_circuit(self, circuit: QuantumCircuit) -> CircuitCharacteristics:
        """
        Analyze circuit and extract relevant characteristics.

        Args:
            circuit: Quantum circuit to analyze

        Returns:
            CircuitCharacteristics with all metrics
        """
        dag = circuit_to_dag(circuit)
        two_qubit_ops = dag.two_qubit_ops()

        # Count gate types
        total_gates = circuit.size()
        two_qubit_gates = len(two_qubit_ops)
        single_qubit_gates = total_gates - two_qubit_gates

        # Count specific gate types
        cnot_gates = sum(
            1 for op in two_qubit_ops
            if op.op.name in ['cx', 'cnot']
        )

        return CircuitCharacteristics(
            num_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            total_gates=total_gates,
            two_qubit_gates=two_qubit_gates,
            cnot_gates=cnot_gates,
            single_qubit_gates=single_qubit_gates,
            circuit=circuit  # Store the actual circuit object
        )


# ============================================================================
# Main QDreamer Class
# ============================================================================

class QDreamerCircuitCutting(Motif):
    """
    Main QDreamer orchestrator for resource-aware circuit cutting optimization.

    Coordinates hardware detection, circuit analysis, and cut optimization.
    
    Example:
        >>> # Basic usage
        >>> qdreamer = QDreamerCircuitCutting(executor, circuit)
        >>> allocation = qdreamer.optimize()
        >>> print(f"Use {allocation.subcircuit_size}q subcircuits for {allocation.speedup_factor:.1f}x speedup")
        
        >>> # With custom calibrated estimator
        >>> estimator = PowerLawEstimator()
        >>> estimator.calibrate_from_measurements(my_measurements)
        >>> qdreamer = QDreamerCircuitCutting(executor, circuit, estimator=estimator)
    """

    def __init__(
        self,
        executor,
        circuit: Optional[QuantumCircuit] = None,
        num_samples: int = 10,
        seed: int = 111,
        use_gpu: Optional[bool] = None,
        estimator: Optional[SpeedupEstimator] = None,
    ):
        """
        Initialize QDreamer Circuit Cutting optimizer.

        Args:
            executor: Executor instance for distributed computation
            circuit: Optional quantum circuit to optimize
            num_samples: Number of quasi-probability samples for circuit cutting
            seed: Random seed for reproducible cut-finding
            use_gpu: Force GPU (True), CPU (False), or auto-detect (None)
            estimator: Custom SpeedupEstimator (default: PowerLawEstimator)
        """
        super().__init__(executor, None)

        self.executor = executor
        self.num_samples = num_samples
        self.seed = seed
        self.use_gpu = use_gpu
        self.estimator = estimator or EstimatorRegistry.get_default()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        executor_config = getattr(executor, 'cluster_config', None)
        self.resource_detector = ResourceDetector(executor_config)
        self.circuit_analyzer = CircuitAnalyzer()

        # Cache
        self._resource_profile: Optional[ResourceProfile] = None
        self._circuit_chars: Optional[CircuitCharacteristics] = None
        self._optimizer: Optional[ResourceOptimizer] = None

        # Circuit
        self.circuit = None
        if circuit is not None:
            self.circuit = circuit

    @property
    def resource_profile(self) -> ResourceProfile:
        """Hardware resource profile (cached)."""
        if self._resource_profile is None:
            self._resource_profile = self.resource_detector.get_local_resources()
        return self._resource_profile

    def analyze(self, circuit: Optional[QuantumCircuit] = None) -> CircuitCharacteristics:
        """Analyze circuit characteristics."""
        if circuit is not None:
            self.circuit = circuit
        if self.circuit is None:
            raise ValueError("No circuit provided")
        
        if self._circuit_chars is None or circuit is not None:
            self._circuit_chars = self.circuit_analyzer.analyze_circuit(self.circuit)
        return self._circuit_chars

    def optimize(self, circuit: Optional[QuantumCircuit] = None) -> OptimizedAllocation:
        """
        Find optimal cutting configuration for the circuit.
        
        Args:
            circuit: Circuit to optimize (uses self.circuit if not provided)
            
        Returns:
            OptimizedAllocation with best configuration
        """
        if circuit is not None:
            self.circuit = circuit
        if self.circuit is None:
            raise ValueError("No circuit provided")
        
        circuit_chars = self.analyze()
        
        self._optimizer = ResourceOptimizer(
            resource_profile=self.resource_profile,
            num_samples=self.num_samples,
            circuit=self.circuit,
            use_gpu=self.use_gpu,
            estimator=self.estimator,
        )
        
        return self._optimizer.find_best_configuration(circuit_chars, seed=self.seed)

    def get_all_configurations(self) -> List[CutConfiguration]:
        """Get all configurations evaluated in the last optimization."""
        if self._optimizer is None:
            return []
        return self._optimizer.get_all_configurations()

    def evaluate_subcircuit_size(
        self,
        subcircuit_size: int,
        circuit: Optional[QuantumCircuit] = None,
    ) -> OptimizedAllocation:
        """
        Evaluate a specific subcircuit size and return the allocation.
        
        Use this when you want to test a specific configuration rather than
        letting optimize() find the best one automatically. Useful for:
        - Calibration experiments comparing predicted vs actual performance
        - Testing specific cutting configurations
        - Benchmarking different subcircuit sizes
        
        Args:
            subcircuit_size: Number of qubits per subcircuit to evaluate
            circuit: Circuit to evaluate (uses self.circuit if not provided)
            
        Returns:
            OptimizedAllocation for the specified subcircuit size
            
        Example:
            >>> qdreamer = QDreamerCircuitCutting(executor, circuit)
            >>> # Evaluate specific sizes for calibration
            >>> for size in [18, 15, 12, 9]:
            ...     allocation = qdreamer.evaluate_subcircuit_size(size)
            ...     print(f"{size}q: {allocation.speedup_factor:.2f}x speedup")
        """
        if circuit is not None:
            self.circuit = circuit
        if self.circuit is None:
            raise ValueError("No circuit provided")
        
        circuit_chars = self.analyze()
        
        # Initialize or reuse optimizer
        if self._optimizer is None:
            self._optimizer = ResourceOptimizer(
                resource_profile=self.resource_profile,
                num_samples=self.num_samples,
                circuit=self.circuit,
                use_gpu=self.use_gpu,
                estimator=self.estimator,
            )
        
        # Evaluate the specific subcircuit size
        config = self._optimizer._evaluate_subcircuit_size(
            num_qubits=circuit_chars.num_qubits,
            subcircuit_size=subcircuit_size,
            seed=self.seed,
        )
        
        # Convert to allocation and return
        return self._optimizer._to_allocation(config, circuit_chars)
    
    def calibrate_estimator(self, measurements: List[Dict]) -> Dict:
        """
        Calibrate the speedup estimator from experimental measurements.
        
        Args:
            measurements: List of dicts with keys:
                n (total qubits), n_sub (subcircuit qubits),
                k (num cuts), W (workers), S (observed speedup)
                
        Returns:
            Dict with fitted parameters (eta_max, p, n_points)
            
        Example:
            >>> measurements = [
            ...     {"n": 36, "n_sub": 20, "k": 2, "W": 8, "S": 520.0},
            ...     {"n": 36, "n_sub": 17, "k": 4, "W": 8, "S": 12.0},
            ... ]
            >>> params = qdreamer.calibrate_estimator(measurements)
            >>> print(f"Calibrated: eta_max={params['eta_max']:.3f}")
        """
        if isinstance(self.estimator, PowerLawEstimator):
            return self.estimator.calibrate_from_measurements(measurements)
        raise TypeError(f"Estimator {type(self.estimator)} does not support calibration")
