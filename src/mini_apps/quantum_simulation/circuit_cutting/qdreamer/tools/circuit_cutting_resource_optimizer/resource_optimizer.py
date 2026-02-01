"""
Resource Optimizer Module for QDreamer

Finds optimal circuit cutting configurations by searching over subcircuit sizes
and using Qiskit's circuit cutting addon to determine actual cut placements.

Author: QDreamer Team
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_cutting.automated_cut_finding import (
    DeviceConstraints,
    OptimizationParameters,
    find_cuts,
)
from qiskit_addon_cutting import (
    cut_wires,
    expand_observables,
    generate_cutting_experiments,
    partition_problem,
)

from ...core.data_models import (
    ResourceProfile,
    CircuitCharacteristics,
    OptimizedAllocation,
    CutConfiguration,
    EstimatorInput,
)
from .estimators import (
    SpeedupEstimator,
    PowerLawEstimator,
    EstimatorRegistry,
)


logger = logging.getLogger(__name__)


class ResourceOptimizer:
    """
    Finds optimal circuit cutting configurations for a given circuit and hardware.
    
    Example:
        >>> optimizer = ResourceOptimizer(resource_profile, circuit=my_circuit)
        >>> best = optimizer.find_best_configuration()
        >>> print(f"Use {best.subcircuit_size}q subcircuits for {best.speedup_factor:.1f}x speedup")
        
        >>> # With custom calibrated estimator
        >>> estimator = PowerLawEstimator()
        >>> estimator.calibrate_from_measurements(my_measurements)
        >>> optimizer = ResourceOptimizer(resource_profile, estimator=estimator)
    """

    def __init__(
        self,
        resource_profile: ResourceProfile,
        num_samples: int = 10000,
        circuit: Optional[any] = None,
        use_gpu: Optional[bool] = None,
        estimator: Optional[SpeedupEstimator] = None,
        estimator_name: str = "power_law",
    ):
        """
        Initialize ResourceOptimizer.

        Args:
            resource_profile: Hardware resource profile (GPUs, CPUs, memory)
            num_samples: Number of quasi-probability samples for Qiskit circuit cutting
            circuit: Optional QuantumCircuit for accurate cut-finding (uses dummy if None)
            use_gpu: Force GPU (True), CPU (False), or auto-detect (None)
            estimator: Custom SpeedupEstimator instance (overrides estimator_name)
            estimator_name: Name of estimator from registry (default: "power_law")
        """
        self.resource_profile = resource_profile
        self.num_samples = num_samples
        self.circuit = circuit
        self.use_gpu = use_gpu if use_gpu is not None else (resource_profile.total_gpus > 0)
        
        # Use provided estimator or get from registry
        self.estimator = estimator or EstimatorRegistry.get(estimator_name)
        
        # Results cache
        self._configurations: List[CutConfiguration] = []

    @property
    def num_workers(self) -> int:
        """Number of parallel workers based on GPU/CPU setting."""
        if self.use_gpu:
            return max(1, self.resource_profile.total_gpus)
        return max(1, self.resource_profile.total_cpus)

    def find_best_configuration(
        self,
        circuit_chars: Optional[CircuitCharacteristics] = None,
        seed: int = 111,
        quick: bool = True,
    ) -> OptimizedAllocation:
        """
        Search for the optimal cutting configuration that maximizes speedup.
        
        Args:
            circuit_chars: Circuit characteristics (optional - computed from circuit if not provided)
            seed: Random seed for cut-finding reproducibility
            quick: If True (default), use fast heuristic estimation.
                   If False, use Qiskit circuit cutting for accurate cut placement.
            
        Returns:
            OptimizedAllocation with the best configuration found.
            If no cutting configuration provides speedup > 1.0, returns a 
            "no cutting" recommendation (num_cuts=0, speedup=1.0).
        """
        # Get num_qubits from circuit_chars or directly from circuit
        if circuit_chars is not None:
            num_qubits = circuit_chars.num_qubits
        elif self.circuit is not None:
            num_qubits = self.circuit.num_qubits
        else:
            raise ValueError("Either circuit_chars or circuit must be provided")
        max_size = self._max_subcircuit_size(num_qubits)
        # Search down to 1/4 of circuit size (but at least 10 qubits for simulation efficiency)
        min_size = max(num_qubits // 4, 10)
        
        if min_size >= max_size:
            min_size = max(max_size - 4, 10)
        
        self._configurations = []
        best: Optional[CutConfiguration] = None
        
        logger.info(f"Searching subcircuit sizes {min_size} to {max_size} for {num_qubits}q circuit")
        
        for size in range(max_size, min_size - 1, -1):
            config = self._evaluate_subcircuit_size(num_qubits, size, seed, quick)
            self._configurations.append(config)
            
            if best is None or config.speedup_factor > best.speedup_factor:
                best = config
                logger.info(f"  NEW BEST: {size}q → {config.speedup_factor:.2f}x speedup ({config.num_cuts} cuts)")
        
        # If the best cutting configuration doesn't beat the "no cutting" baseline (1.0x),
        # return a "no cutting" recommendation instead
        if best.speedup_factor < 1.0:
            logger.info(f"  No beneficial cutting found (best was {best.speedup_factor:.2f}x). Recommending no cuts.")
            no_cut_config = CutConfiguration(
                subcircuit_size=num_qubits,
                num_cuts=0,
                num_tasks=1,
                sampling_overhead=1.0,
                speedup_factor=1.0,
                metadata={'reason': 'no_cutting_beneficial', 'best_cutting_speedup': best.speedup_factor}
            )
            return self._to_allocation(no_cut_config)
        
        return self._to_allocation(best)

    def get_all_configurations(self) -> List[CutConfiguration]:
        """Get all configurations evaluated in the last search."""
        return self._configurations

    def _evaluate_subcircuit_size(
        self,
        num_qubits: int,
        subcircuit_size: int,
        seed: int,
        quick: bool = True,
    ) -> CutConfiguration:
        """Evaluate a specific subcircuit size.
        
        Args:
            quick: If True, use heuristic estimation. If False, use Qiskit cut-finding.
        """
        if quick:
            cut_info, num_tasks = self._estimate_cuts(num_qubits, subcircuit_size)
        else:
            try:
                cut_info, num_tasks = self._run_qiskit_cutting(num_qubits, subcircuit_size, seed)
            except Exception as e:
                logger.warning(f"Cut-finding failed for {subcircuit_size}q: {e}")
                cut_info, num_tasks = self._estimate_cuts(num_qubits, subcircuit_size)
        
        # Estimate speedup
        result = self.estimator.estimate_speedup(EstimatorInput(
            total_qubits=num_qubits,
            subcircuit_qubits=subcircuit_size,
            num_cuts=cut_info['num_cuts'],
            num_workers=self.num_workers,
            num_tasks=num_tasks,
        ))
        
        return CutConfiguration(
            subcircuit_size=subcircuit_size,
            num_cuts=cut_info['num_cuts'],
            num_tasks=num_tasks,
            sampling_overhead=cut_info['sampling_overhead'],
            speedup_factor=result.speedup_factor,
            metadata=cut_info.get('metadata', {}),
        )

    def _run_qiskit_cutting(
        self,
        num_qubits: int,
        subcircuit_size: int,
        seed: int,
    ) -> Tuple[Dict, int]:
        """Run Qiskit circuit cutting to get actual cuts and task count."""
        circuit = self._get_circuit(num_qubits)
        
        cut_circuit, metadata = find_cuts(
            circuit,
            OptimizationParameters(seed=seed),
            DeviceConstraints(qubits_per_subcircuit=subcircuit_size),
        )
        
        # Generate experiments to get actual task count
        observable = SparsePauliOp("Z" * num_qubits)
        qc_w_ancilla = cut_wires(cut_circuit)
        observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)
        partitioned = partition_problem(circuit=qc_w_ancilla, observables=observables_expanded)
        
        subexperiments, _ = generate_cutting_experiments(
            circuits=partitioned.subcircuits,
            observables=partitioned.subobservables,
            num_samples=self.num_samples,
        )
        
        num_tasks = sum(len(expts) for expts in subexperiments.values())
        num_cuts = len(metadata.get('cuts', []))
        
        return {
            'num_cuts': num_cuts,
            'sampling_overhead': metadata.get('sampling_overhead', 9.0 ** num_cuts),
            'metadata': {**metadata, 'optimization_seed': seed},
        }, num_tasks

    def _estimate_cuts(self, num_qubits: int, subcircuit_size: int) -> Tuple[Dict, int]:
        """Estimate cuts based on subcircuit size (for linear entanglement circuits).
        
        For a circuit with linear entanglement, the number of cuts needed is
        (number of partitions - 1), where partitions = ceil(n / n_sub).
        """
        num_partitions = math.ceil(num_qubits / subcircuit_size)
        num_cuts = max(1, num_partitions - 1)
        num_tasks = 9 ** num_cuts
        return {
            'num_cuts': num_cuts,
            'sampling_overhead': 9.0 ** num_cuts,
            'metadata': {},
        }, num_tasks

    def _get_circuit(self, num_qubits: int):
        """Get circuit for cut-finding (user circuit or dummy)."""
        if self.circuit is not None and self.circuit.num_qubits == num_qubits:
            return self.circuit
        logger.debug(f"Using dummy EfficientSU2 circuit ({num_qubits}q)")
        return EfficientSU2(num_qubits, entanglement='linear', reps=2).decompose()

    def _max_subcircuit_size(self, num_qubits: int) -> int:
        """Calculate max subcircuit size based on available memory."""
        if self.use_gpu and self.resource_profile.gpu_memory_mb:
            mem_mb = self.resource_profile.gpu_memory_mb[0] * 0.8
            max_qubits = int(np.log2(mem_mb * 1024 * 1024 / 16))
        else:
            mem_gb = self.resource_profile.available_memory_gb * 0.8
            max_qubits = int(np.log2(mem_gb * 1024 * 1024 * 1024 / 16))
        return min(max_qubits, num_qubits - 1, 32)

    def _to_allocation(self, config: CutConfiguration) -> OptimizedAllocation:
        """Convert CutConfiguration to OptimizedAllocation."""
        return OptimizedAllocation(
            subcircuit_size=config.subcircuit_size,
            num_cuts=config.num_cuts,
            num_parallel_tasks=config.num_tasks,
            use_gpu=self.use_gpu,
            sampling_overhead=config.sampling_overhead,
            metadata=config.metadata,
            speedup_factor=config.speedup_factor,
        )
