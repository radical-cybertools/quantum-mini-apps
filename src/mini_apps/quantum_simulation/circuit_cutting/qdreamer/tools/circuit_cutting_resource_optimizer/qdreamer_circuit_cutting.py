"""
QDreamer Main Class

The main QDreamerCircuitCutting orchestrator for resource-aware circuit cutting optimization.

Author: QDreamer Team
"""

import logging
from typing import Dict, List, Optional

from qiskit import QuantumCircuit

from engine.base.base_motif import Motif
from ...core.data_models import (
    ResourceProfile,
    CircuitCharacteristics,
    OptimizedAllocation,
    CutConfiguration,
)
from .resource_optimizer import ResourceOptimizer
from ...core.detector import ResourceDetector, CircuitAnalyzer
from .estimators import SpeedupEstimator, PowerLawEstimator, EstimatorRegistry


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

    def optimize(
        self,
        circuit: Optional[QuantumCircuit] = None,
        quick: bool = True,
    ) -> OptimizedAllocation:
        """
        Find optimal cutting configuration for the circuit.
        
        Args:
            circuit: Circuit to optimize (uses self.circuit if not provided)
            quick: If True (default), use fast heuristic estimation.
                   If False, use Qiskit circuit cutting for accurate cut placement.
            
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
        
        return self._optimizer.find_best_configuration(circuit_chars, seed=self.seed, quick=quick)

    def get_all_configurations(self) -> List[CutConfiguration]:
        """Get all configurations evaluated in the last optimization."""
        if self._optimizer is None:
            return []
        return self._optimizer.get_all_configurations()

    def evaluate_subcircuit_size(
        self,
        subcircuit_size: int,
        circuit: Optional[QuantumCircuit] = None,
        quick: bool = True,
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
            quick: If True (default), use fast heuristic estimation.
                   If False, use Qiskit circuit cutting for accurate cut placement.
            
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
            quick=quick,
        )
        
        # Convert to allocation and return
        return self._optimizer._to_allocation(config)
    
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
