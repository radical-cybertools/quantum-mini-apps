"""
QDreamer Estimators Module

Provides speedup estimation plugins for circuit cutting optimization:
- SpeedupEstimator: Abstract base class
- PowerLawEstimator: Default power-law efficiency decay model
- EstimatorRegistry: Plugin registry
"""

from .base import SpeedupEstimator
from .power_law import (
    PowerLawEstimator,
    EstimatorRegistry,
    fit_efficiency_power_law,
)

__all__ = [
    'SpeedupEstimator',
    'PowerLawEstimator',
    'EstimatorRegistry',
    'fit_efficiency_power_law',
]
