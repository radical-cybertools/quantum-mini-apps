"""
Speedup Prediction Model for Circuit Cutting

This module implements the QDreamer speedup prediction model for parallel circuit cutting
versus sequential execution. The model accounts for:
- Subcircuit execution time
- Parallel task distribution
- Reconstruction overhead
- Amdahl's law limitations
"""

import numpy as np
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass


@dataclass
class HardwareCalibration:
    """
    Hardware-specific calibration parameters for the speedup prediction model.
    
    These parameters are calibrated empirically on production hardware using
    EfficientSU2 benchmark circuits.
    
    Attributes:
        c_h: Framework overhead constant that corrects for hardware-specific overhead
        e_h: Parallel efficiency factor accounting for resource contention and coordination costs
        r_h: Reconstruction overhead constant for quasi-probability reconstruction
        s_max: Maximum achievable speedup accounting for Amdahl's law limitations
    """
    c_h: float = 1.0
    e_h: float = 1.0
    r_h: float = 1.0
    s_max: float = float('inf')


class SpeedupPredictor:
    """
    Predicts the speedup of parallel circuit cutting versus sequential execution.
    
    The model predicts speedup S using:
        S = min(1/T_total, S_max)
    
    where T_total = t_par + t_rec (parallel execution time + reconstruction time).
    """
    
    def __init__(
        self,
        calibration: Optional[HardwareCalibration] = None,
        sampling_base: float = 9.0
    ):
        """
        Initialize the speedup predictor.
        
        Args:
            calibration: Hardware calibration parameters. If None, uses default values.
            sampling_base: Sampling base for reconstruction (typically 9 for gate cuts).
        """
        self.calibration = calibration or HardwareCalibration()
        self.sampling_base = sampling_base
    
    def compute_subcircuit_time(
        self,
        n: int,
        n_sub: int
    ) -> float:
        """
        Compute the relative execution time for a single subcircuit.
        
        Equation (3): t_sub = 2^(n_sub - n) · c_h
        
        Args:
            n: Total number of qubits in the original circuit
            n_sub: Number of qubits in the subcircuit
            
        Returns:
            Relative execution time for a single subcircuit
        """
        return (2 ** (n_sub - n))                                                                                                                                         
    
    def compute_parallel_time(
        self,
        n: int,
        n_sub: int,
        M: int,
        W: int,
        M_seq: Optional[int] = None
    ) -> float:
        """
        Compute the parallel execution time with efficiency correction.
        
        If M_seq is provided, models parallel subcircuits with sequential subexperiments:
        - M = number of subcircuits (parallel units)
        - M_seq = max subexperiments per subcircuit (sequential within each subcircuit)
        - t_par = ⌈M/W⌉ · M_seq · t_sub / e_h
        
        Otherwise, uses original model:
        - M = number of subexperiments
        - t_par = ⌈M/W⌉ · t_sub / e_h
        
        Args:
            n: Total number of qubits in the original circuit
            n_sub: Number of qubits in the subcircuit
            M: Number of subcircuits (if M_seq provided) or subexperiments (otherwise)
            W: Number of workers (GPUs or CPU cores)
            M_seq: Optional. Max number of subexperiments per subcircuit (sequential within subcircuit)
            
        Returns:
            Parallel execution time
        """
        t_sub = self.compute_subcircuit_time(n, n_sub)
        batches = np.ceil(M / W)
        
        if M_seq is not None:
            # Model: subcircuits run in parallel, subexperiments within each subcircuit run sequentially
            # Time per batch = max(M_seq) * t_sub (worst case subcircuit)
            return (batches * M_seq * t_sub) / self.calibration.e_h
        else:
            # Original model: all subexperiments are parallel units
            return (batches * t_sub) / self.calibration.e_h
    
    def compute_reconstruction_time(
        self,
        k: int
    ) -> float:
        """
        Compute the reconstruction overhead time.
        
        The quasi-probability reconstruction requires sequential post-processing
        that scales with the number of subexperiments. For a cutting solution
        with k cuts, the number of subexperiments grows exponentially with
        respect to the sampling base.
        
        Equation (5): t_rec = r_h · b^k
        
        Args:
            k: Number of cuts in the cutting solution
            
        Returns:
            Reconstruction time
        """
        return self.calibration.r_h * (self.sampling_base ** k)
    
    def compute_total_time(
        self,
        n: int,
        n_sub: int,
        M: int,
        W: int,
        k: int
    ) -> float:
        """
        Compute the total execution time for parallel circuit cutting.
        
        Equation (2): T_total = t_par + t_rec
        
        Args:
            n: Total number of qubits in the original circuit
            n_sub: Number of qubits in the subcircuit
            M: Number of subexperiments generated by circuit cutting
            W: Number of workers (GPUs or CPU cores)
            k: Number of cuts in the cutting solution
            
        Returns:
            Total execution time
        """
        t_par = self.compute_parallel_time(n, n_sub, M, W)
        t_rec = self.compute_reconstruction_time(k)
        return t_par + t_rec
    
    def predict_speedup(
        self,
        n: int,
        n_sub: int,
        M: int,
        W: int,
        k: int,
        M_seq: Optional[int] = None,
        return_breakdown: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Predict the speedup of parallel circuit cutting versus sequential execution.
        
        All time components are relative (normalized to full circuit execution time):
        - t_sub: relative subcircuit time (normalized)
        - t_par: relative parallel time (normalized)  
        - t_rec: relative reconstruction time (normalized)
        
        Equation (2): T_total = t_par + t_rec (both relative)
        Equation (1): S = min(1/T_total, S_max)
        
        Where:
        - T_total = 1.0 means same time as full circuit (speedup = 1.0)
        - T_total < 1.0 means faster (speedup > 1.0)
        - T_total > 1.0 means slower (speedup < 1.0)
        
        Args:
            n: Total number of qubits in the original circuit
            n_sub: Number of qubits in the subcircuit
            M: Number of subcircuits (if M_seq provided) or subexperiments (otherwise)
            W: Number of workers (GPUs or CPU cores)
            k: Number of cuts in the cutting solution
            M_seq: Optional. Max number of subexperiments per subcircuit (sequential within subcircuit)
            return_breakdown: If True, returns a dictionary with detailed breakdown.
                             If False, returns just the speedup factor (float).
            
        Returns:
            If return_breakdown=False: Predicted speedup factor (float)
            If return_breakdown=True: Dictionary containing:
                - speedup: Predicted speedup factor
                - total_time: Total relative time (T_total = t_par + t_rec)
                - parallel_time: Parallel execution time (t_par, relative)
                - reconstruction_time: Reconstruction time (t_rec, relative)
                - subcircuit_time: Single subcircuit execution time (t_sub, relative)
                - batches: Number of batches (⌈M/W⌉)
                - num_subexperiments: M
                - num_workers: W
                - num_cuts: k
        """
        t_sub = self.compute_subcircuit_time(n, n_sub)
        t_par = self.compute_parallel_time(n, n_sub, M, W, M_seq)
        t_rec = self.compute_reconstruction_time(k)
        T_total = t_par + t_rec
        
        # Compute speedup: S = 1/T_total (since T_total is relative)
        # T_total = 1.0 means same time as full circuit, so speedup = 1.0
        # T_total < 1.0 means faster, so speedup > 1.0
        if T_total > 0:
            speedup = min(1.0 / T_total, self.calibration.s_max)
        else:
            speedup = self.calibration.s_max
        
        if not return_breakdown:
            return speedup
        
        batches = int(np.ceil(M / W))
        breakdown = {
            'speedup': speedup,
            'total_time': T_total,
            'parallel_time': t_par,
            'reconstruction_time': t_rec,
            'subcircuit_time': t_sub,
            'batches': batches,
            'num_subcircuits': M if M_seq is not None else None,
            'num_subexperiments': M if M_seq is None else None,
            'max_subexperiments_per_subcircuit': M_seq,
            'num_workers': W,
            'num_cuts': k
        }
        return breakdown


# Example usage and default calibrations
def get_default_cpu_calibration() -> HardwareCalibration:
    """Get default calibration parameters for CPU hardware."""
    return HardwareCalibration(
        c_h=1.0,  # Should be calibrated empirically
        e_h=0.8,  # Typical parallel efficiency
        r_h=0.001,  # Should be calibrated empirically
        s_max=100.0  # Reasonable upper bound
    )


def get_default_gpu_calibration() -> HardwareCalibration:
    """Get default calibration parameters for GPU hardware."""
    return HardwareCalibration(
        c_h=0.5,  # GPUs typically have lower overhead
        e_h=0.7,  # GPU parallel efficiency (may be lower due to memory transfer)
        r_h=0.0005,  # Should be calibrated empirically
        s_max=1000.0  # Higher potential speedup with GPUs
    )



