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

## Usage

```
python qml_compression.py --num_nodes 1
```



