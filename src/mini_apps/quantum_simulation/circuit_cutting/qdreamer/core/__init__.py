"""
QDreamer Core Module

Contains shared core components:
- Data models (ResourceProfile, OptimizedAllocation, etc.)
- Resource detector and circuit analyzer
"""

from .data_models import (
    ResourceProfile,
    CircuitCharacteristics,
    OptimizedAllocation,
    CutConfiguration,
    EstimatorInput,
    SpeedupResult,
)

from .detector import (
    ResourceDetector,
    CircuitAnalyzer,
)

__all__ = [
    # Data models
    'ResourceProfile',
    'CircuitCharacteristics',
    'OptimizedAllocation',
    'CutConfiguration',
    'EstimatorInput',
    'SpeedupResult',
    # Detector/Analyzer
    'ResourceDetector',
    'CircuitAnalyzer',
]
