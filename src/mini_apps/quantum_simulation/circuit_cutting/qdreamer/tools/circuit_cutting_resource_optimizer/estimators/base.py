"""
Speedup Estimator Base Class

Abstract base class defining the interface for speedup estimation plugins.

Author: QDreamer Team
"""

from abc import ABC, abstractmethod

from ....core.data_models import EstimatorInput, SpeedupResult


class SpeedupEstimator(ABC):
    """
    Abstract base class for speedup estimators.
    
    All speedup estimation plugins must inherit from this class and implement
    the estimate_speedup() method and name property.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this estimator."""
        pass
    
    @abstractmethod
    def estimate_speedup(self, config: EstimatorInput) -> SpeedupResult:
        """
        Estimate speedup for the given configuration.
        
        Args:
            config: EstimatorInput with circuit and resource parameters
            
        Returns:
            SpeedupResult with estimated speedup metrics
        """
        pass
