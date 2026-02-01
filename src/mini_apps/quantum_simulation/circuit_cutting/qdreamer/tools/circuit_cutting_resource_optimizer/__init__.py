"""
Circuit Cutting Resource Optimizer

Resource-aware optimization for quantum circuit cutting that automatically detects
hardware resources and finds optimal circuit partitioning configurations.

Components:
- QDreamerCircuitCutting: Main orchestrator with executor integration
- ResourceOptimizer: Standalone optimizer (no executor needed)
- estimators: Speedup estimation plugins (power-law, etc.)
"""

from .qdreamer_circuit_cutting import QDreamerCircuitCutting
from .resource_optimizer import ResourceOptimizer

from .estimators import (
    SpeedupEstimator,
    PowerLawEstimator,
    EstimatorRegistry,
    fit_efficiency_power_law,
)

__all__ = [
    'QDreamerCircuitCutting',
    'ResourceOptimizer',
    'SpeedupEstimator',
    'PowerLawEstimator',
    'EstimatorRegistry',
    'fit_efficiency_power_law',
]
