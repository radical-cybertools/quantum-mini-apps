"""
QDreamer v3.0: Resource-Aware Quantum Circuit Cutting Optimization

Finds optimal circuit cutting configurations based on hardware resources.

Quick Start:
    >>> from qdreamer import QDreamerCircuitCutting, predict_cutting
    >>> 
    >>> # Fast prediction (no Qiskit needed)
    >>> result = predict_cutting(n_qubits=36, num_workers=8)
    >>> print(f"Speedup: {result['speedup']:.1f}x with {result['subcircuit_size']}q subcircuits")
    >>> 
    >>> # Full optimization with actual circuit
    >>> qdreamer = QDreamerCircuitCutting(executor, circuit)
    >>> allocation = qdreamer.optimize()

Estimator Calibration:
    >>> from qdreamer import PowerLawEstimator
    >>> 
    >>> estimator = PowerLawEstimator()
    >>> estimator.calibrate_from_measurements([
    ...     {"n": 36, "n_sub": 20, "k": 2, "W": 8, "S": 520.0},
    ...     {"n": 36, "n_sub": 17, "k": 4, "W": 8, "S": 12.0},
    ... ])
    >>> 
    >>> qdreamer = QDreamerCircuitCutting(executor, circuit, estimator=estimator)
"""

from .qdreamer import (
    QDreamerCircuitCutting,
    ResourceDetector,
    CircuitAnalyzer,
)

from .resource_optimizer import (
    ResourceOptimizer,
    predict_cutting,
)

from .data_models import (
    ResourceProfile,
    CircuitCharacteristics,
    OptimizedAllocation,
    CutConfiguration,
    EstimatorInput,
    SpeedupResult,
)

from .estimators import (
    SpeedupEstimator,
    PowerLawEstimator,
    EstimatorRegistry,
    fit_efficiency_power_law,
)

__all__ = [
    # Main API
    'QDreamerCircuitCutting',
    'ResourceOptimizer',
    'predict_cutting',
    
    # Estimators
    'SpeedupEstimator',
    'PowerLawEstimator',
    'EstimatorRegistry',
    'fit_efficiency_power_law',
    
    # Data classes
    'ResourceProfile',
    'CircuitCharacteristics',
    'OptimizedAllocation',
    'CutConfiguration',
    'EstimatorInput',
    'SpeedupResult',
    
    # Utilities
    'ResourceDetector',
    'CircuitAnalyzer',
]

__version__ = '0.2.0'
