# QML Data Compression Mini-App

This mini-app implements a quantum machine learning approach for data compression using quantum circuits and Matrix Product States (MPS). It demonstrates how quantum computing techniques can be applied to classical data compression tasks, potentially offering advantages in compression ratios and processing efficiency.

## Overview

The QML Data Compression mini-app performs the following steps:
1. Encodes classical image data into quantum states using FRQI (Flexible Representation of Quantum Images)
2. Transforms the quantum states into Matrix Product States (MPS)
3. Fits a staircase quantum circuit to the target MPS
4. Optimizes the quantum circuit parameters using BFGS (Broyden–Fletcher–Goldfarb–Shanno algorithm)

## Technical Background

The mini-app leverages several quantum computing concepts:

- **FRQI Encoding**: Converts classical image pixels into quantum states while preserving spatial and color information
- **Matrix Product States**: Represents quantum many-body states efficiently using tensor networks
- **Quantum Circuit Optimization**: Uses variational quantum algorithms to find optimal circuit parameters

## Requirements

### Software Dependencies
- Python 3.10
- PennyLane >= 0.30.0
- NumPy >= 1.21.0
- PyYAML >= 6.0.0
- Ray >= 2.3.0 (for distributed computing)
- Matplotlib >= 3.5.0 (for visualization)
- SciPy >= 1.7.0

### Hardware Requirements
- SLURM-based HPC environment
- Minimum 16GB RAM per node
- High-speed interconnect for distributed computing

## Usage





