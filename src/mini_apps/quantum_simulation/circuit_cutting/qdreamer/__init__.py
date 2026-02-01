"""
QDreamer v0.4.0: Resource-Aware Quantum Circuit Cutting Optimization

Finds optimal circuit cutting configurations based on hardware resources.

Quick Start:
    >>> from qdreamer import QDreamerCircuitCutting
    >>> 
    >>> # Optimization with circuit (quick=True by default for fast heuristic)
    >>> qdreamer = QDreamerCircuitCutting(executor, circuit)
    >>> allocation = qdreamer.optimize()
    >>> 
    >>> # For more accurate Qiskit-based cut finding, use quick=False
    >>> allocation = qdreamer.optimize(quick=False)

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

Module Structure:
    qdreamer/
    ├── core/           - Shared data models and detectors
    ├── tools/          - Optimization tools
    │   └── circuit_cutting_resource_optimizer/
    │       ├── resource_optimizer.py
    │       ├── qdreamer_circuit_cutting.py
    │       └── estimators/
    └── examples/       - Example scripts (basic, executor, calibration)
"""

# Core data models and utilities
from .core import (
    ResourceDetector,
    CircuitAnalyzer,
    ResourceProfile,
    CircuitCharacteristics,
    OptimizedAllocation,
    CutConfiguration,
    EstimatorInput,
    SpeedupResult,
)

# Tools - Circuit Cutting Resource Optimizer
from .tools.circuit_cutting_resource_optimizer import (
    QDreamerCircuitCutting,
    ResourceOptimizer,
    SpeedupEstimator,
    PowerLawEstimator,
    EstimatorRegistry,
    fit_efficiency_power_law,
)

__all__ = [
    # Main API
    'QDreamerCircuitCutting',
    'ResourceOptimizer',
    
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

__version__ = '0.4.0'
