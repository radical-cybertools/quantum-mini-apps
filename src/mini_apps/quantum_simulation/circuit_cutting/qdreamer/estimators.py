"""
Speedup Estimator Plugin API for QDreamer

This module provides a plugin-based architecture for speedup estimation in circuit cutting.
It includes:
- SpeedupEstimator: Abstract base class defining the estimator interface
- PowerLawEstimator: Default estimator using power-law efficiency decay model
- EstimatorRegistry: Global registry for plugin discovery and selection

The power-law model is based on the equation: η(R) = η_max / R^p
where R is the number of execution rounds and p is the decay exponent.

Author: QDreamer Team
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data_models import EstimatorInput, SpeedupResult


logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Class
# =============================================================================


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


# =============================================================================
# Power-Law Efficiency Model
# =============================================================================


class PowerLawEstimator(SpeedupEstimator):
    """
    Speedup estimator using power-law efficiency decay model.
    
    The model predicts speedup based on:
    - Full circuit cost: C_full = 2^n
    - Cutting cost: C_cut = 2^n_sub × R / η(R)
    - Efficiency decay: η(R) = η_max / R^p
    - Speedup: S = C_full / C_cut = 2^(n - n_sub) × η(R) / R
    
    Default parameters (eta_max=0.0008, decay_exponent=0.3233) are fitted from
    experimental measurements on GPU clusters.
    
    Example:
        >>> estimator = PowerLawEstimator(eta_max=0.0008, decay_exponent=0.3233)
        >>> config = EstimatorInput(
        ...     total_qubits=36,
        ...     subcircuit_qubits=20,
        ...     num_cuts=2,
        ...     num_workers=8,
        ... )
        >>> result = estimator.estimate_speedup(config)
        >>> print(f"Speedup: {result.speedup_factor:.1f}x")
    """
    
    def __init__(
        self,
        eta_max: float = 0.0008,
        decay_exponent: float = 0.3233,
    ):
        """
        Initialize PowerLawEstimator.
        
        Args:
            eta_max: Peak parallel efficiency at R=1 round (default: 0.0008)
            decay_exponent: Power-law decay exponent p (default: 0.3233)
        """
        self._eta_max = float(eta_max)
        self._decay_exponent = float(decay_exponent)
    
    @property
    def name(self) -> str:
        return "power_law"
    
    @property
    def eta_max(self) -> float:
        """Peak parallel efficiency parameter."""
        return self._eta_max
    
    @property
    def decay_exponent(self) -> float:
        """Efficiency decay exponent parameter."""
        return self._decay_exponent
    
    def estimate_speedup(self, config: EstimatorInput) -> SpeedupResult:
        """
        Estimate speedup using power-law efficiency model.
        
        Args:
            config: EstimatorInput with circuit and resource parameters
            
        Returns:
            SpeedupResult with estimated metrics
        """
        # Handle no-cutting baseline
        if config.num_cuts == 0:
            return SpeedupResult(speedup_factor=1.0, efficiency=1.0, num_rounds=1)
        
        # Calculate number of tasks and rounds
        num_tasks = config.num_tasks if config.num_tasks else 9 ** config.num_cuts
        num_workers = max(1, config.num_workers)
        num_rounds = math.ceil(num_tasks / num_workers)
        
        # Efficiency: η(R) = η_max / R^p
        efficiency = self._eta_max / (num_rounds ** self._decay_exponent)
        efficiency = float(np.clip(efficiency, 1e-12, 1.0))
        
        # Speedup: S = 2^(n - n_sub) × η / R
        qubit_reduction = config.total_qubits - config.subcircuit_qubits
        speedup = (2.0 ** qubit_reduction) * efficiency / num_rounds
        
        return SpeedupResult(
            speedup_factor=speedup,
            efficiency=efficiency,
            num_rounds=num_rounds,
        )
    
    def calibrate_from_measurements(
        self,
        measurements: Sequence[Dict[str, float]],
        weight_mode: str = "count",
    ) -> Dict[str, float]:
        """
        Calibrate estimator parameters from experimental measurements.
        
        Fits the power-law model (η = η_max / R^p) to observed speedup data
        using weighted least-squares regression in log-log space.
        
        Args:
            measurements: List of measurement dicts with keys:
                Required: n (total qubits), n_sub (subcircuit qubits),
                         k (num cuts), W (workers), S (observed speedup)
                Optional: count (for weighting)
            weight_mode: Weighting scheme ("count", "uniform")
            
        Returns:
            Dict with fitted parameters (eta_max, p, n_points)
            
        Example:
            >>> estimator = PowerLawEstimator()
            >>> measurements = [
            ...     {"n": 36, "n_sub": 20, "k": 2, "W": 8, "S": 520.2, "count": 3},
            ...     {"n": 36, "n_sub": 17, "k": 4, "W": 8, "S": 12.4, "count": 4},
            ... ]
            >>> params = estimator.calibrate_from_measurements(measurements)
            >>> print(f"Fitted: eta_max={params['eta_max']:.3f}, p={params['p']:.3f}")
        """
        fitted = fit_efficiency_power_law(measurements, weight_mode=weight_mode)
        
        self._eta_max = fitted["eta_max"]
        self._decay_exponent = fitted["p"]
        
        logger.info(
            f"PowerLawEstimator calibrated: eta_max={self._eta_max:.4f}, "
            f"p={self._decay_exponent:.4f} (from {fitted['n_points']} measurements)"
        )
        
        return fitted
    
    def __repr__(self) -> str:
        return f"PowerLawEstimator(eta_max={self._eta_max:.4f}, decay_exponent={self._decay_exponent:.4f})"


# =============================================================================
# Calibration
# =============================================================================


def fit_efficiency_power_law(
    measurements: Sequence[Dict[str, float]],
    weight_mode: str = "count",
) -> Dict[str, float]:
    """
    Fit efficiency power-law model: η(R) = η_max / R^p
    
    Uses weighted least-squares regression in log-log space.
    
    Args:
        measurements: List of measurement dicts with keys:
            Required: n, n_sub, k, W, S
            Optional: count (for weighting)
        weight_mode: "count" or "uniform"
        
    Returns:
        Dict with: eta_max, p, n_points
        
    Raises:
        ValueError: If fewer than 2 valid measurements
    """
    regression_data = []
    
    for m in measurements:
        num_cuts = int(m.get("k", 0))
        if num_cuts == 0:
            continue  # Skip baseline measurements
        
        # Extract values
        n, n_sub = int(m["n"]), int(m["n_sub"])
        num_workers = max(1, int(m["W"]))
        speedup = float(m["S"])
        
        # Calculate rounds and implied efficiency
        num_tasks = 9 ** num_cuts
        num_rounds = math.ceil(num_tasks / num_workers)
        
        # S = 2^(n - n_sub) × η / R  =>  η = S × R / 2^(n - n_sub)
        qubit_factor = 2.0 ** (n - n_sub)
        efficiency = speedup * num_rounds / qubit_factor
        
        if efficiency <= 0 or not np.isfinite(efficiency):
            continue
        
        weight = float(m.get("count", 1.0)) if weight_mode == "count" else 1.0
        regression_data.append((math.log(num_rounds), math.log(efficiency), weight))
    
    if len(regression_data) < 2:
        raise ValueError(f"Need at least 2 data points, got {len(regression_data)}")
    
    # Weighted least-squares: log(η) = log(η_max) - p × log(R)
    X = np.array([[1.0, d[0]] for d in regression_data])
    y = np.array([d[1] for d in regression_data])
    w = np.sqrt([d[2] for d in regression_data])
    
    coeffs, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    
    return {
        "eta_max": float(np.exp(coeffs[0])),
        "p": float(-coeffs[1]),
        "n_points": len(regression_data),
    }


# =============================================================================
# Estimator Registry
# =============================================================================


class EstimatorRegistry:
    """
    Global registry for speedup estimator plugins.
    
    Example:
        >>> my_estimator = PowerLawEstimator(eta_max=0.5, decay_exponent=0.4)
        >>> EstimatorRegistry.register("my_custom", my_estimator)
        >>> estimator = EstimatorRegistry.get("my_custom")
    """
    
    _estimators: Dict[str, SpeedupEstimator] = {}
    _default_name: str = "power_law"
    
    @classmethod
    def register(cls, name: str, estimator: SpeedupEstimator) -> None:
        """Register an estimator with the given name."""
        if not isinstance(estimator, SpeedupEstimator):
            raise TypeError(f"Expected SpeedupEstimator, got {type(estimator)}")
        cls._estimators[name] = estimator
    
    @classmethod
    def get(cls, name: str) -> SpeedupEstimator:
        """Get an estimator by name."""
        if name not in cls._estimators:
            raise KeyError(f"Estimator '{name}' not found. Available: {cls.list_available()}")
        return cls._estimators[name]
    
    @classmethod
    def get_default(cls) -> SpeedupEstimator:
        """Get the default estimator."""
        return cls.get(cls._default_name)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered estimator names."""
        return list(cls._estimators.keys())


# Register default estimator on module import
EstimatorRegistry.register("power_law", PowerLawEstimator())


__all__ = [
    "EstimatorInput",
    "SpeedupResult",
    "SpeedupEstimator",
    "PowerLawEstimator",
    "EstimatorRegistry",
    "fit_efficiency_power_law",
]
