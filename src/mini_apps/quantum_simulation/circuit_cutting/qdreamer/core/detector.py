"""
Resource Detection and Circuit Analysis for QDreamer

This module provides:
- ResourceDetector: Detects GPUs, CPUs, and memory
- CircuitAnalyzer: Analyzes quantum circuit characteristics

Author: QDreamer Team
"""

import logging
import subprocess
from typing import Dict, Optional

import psutil
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from .data_models import ResourceProfile, CircuitCharacteristics


class ResourceDetector:
    """
    Detects and profiles local hardware resources including GPUs, CPUs, and memory.
    Supports both single-node and multi-node cluster configurations.
    """

    def __init__(self, executor_config: Optional[Dict] = None):
        """
        Initialize ResourceDetector.

        Args:
            executor_config: Optional executor configuration dict with cluster info.
                           Expected to have 'config' dict with keys like:
                           - 'number_of_nodes': int
                           - 'gpus_per_node': int
                           - 'cores_per_node': int
        """
        self.logger = logging.getLogger(__name__)
        self.executor_config = executor_config

    def get_local_resources(self) -> ResourceProfile:
        """
        Detect all available local resources, considering multi-node cluster config.

        Returns:
            ResourceProfile: Complete profile of local/cluster hardware resources
        """
        profile = ResourceProfile()

        # Detect GPUs
        gpu_info = self._detect_gpus()
        profile.num_gpus = gpu_info['num_gpus']
        profile.gpu_memory_mb = gpu_info['memory_mb']
        profile.gpu_names = gpu_info['names']

        # Detect CPUs
        profile.num_cpu_cores_physical = psutil.cpu_count(logical=False) or 0
        profile.num_cpu_cores_logical = psutil.cpu_count(logical=True) or 0

        # Detect Memory
        mem = psutil.virtual_memory()
        profile.total_memory_gb = mem.total / (1024 ** 3)
        profile.available_memory_gb = mem.available / (1024 ** 3)

        # Apply executor cluster configuration if available
        if self.executor_config and 'config' in self.executor_config:
            config = self.executor_config['config']

            # Get number of nodes (default to 1 if not specified)
            profile.number_of_nodes = config.get('number_of_nodes', 1)

            # Override per-node resources if explicitly specified in config
            if 'gpus_per_node' in config:
                profile.gpus_per_node = config['gpus_per_node']
            else:
                profile.gpus_per_node = profile.num_gpus

            if 'cores_per_node' in config:
                profile.cpus_per_node = config['cores_per_node']
            else:
                profile.cpus_per_node = profile.num_cpu_cores_physical

            self.logger.info(f"Applied multi-node config: {profile.number_of_nodes} nodes")
        else:
            # No executor config - use single node with detected resources
            profile.number_of_nodes = 1
            profile.gpus_per_node = profile.num_gpus
            profile.cpus_per_node = profile.num_cpu_cores_physical

        # Warn if detected GPUs differ from configured GPUs
        if profile.num_gpus > 0 and profile.total_gpus == 0:
            self.logger.warning(
                f"GPUs detected ({profile.num_gpus}) but disabled in configuration "
                f"(gpus_per_node={profile.gpus_per_node}). GPU acceleration will not be used."
            )
        elif profile.num_gpus != profile.total_gpus and profile.num_gpus > 0:
            self.logger.info(
                f"GPU configuration: {profile.num_gpus} detected locally, "
                f"{profile.total_gpus} total across {profile.number_of_nodes} node(s)"
            )

        self.logger.info(f"Detected resources:\n{profile}")
        return profile

    def _detect_gpus(self) -> Dict[str, any]:
        """
        Detect NVIDIA GPUs using nvidia-smi.

        Returns:
            Dict with 'num_gpus', 'memory_mb', and 'names'
        """
        result = {
            'num_gpus': 0,
            'memory_mb': [],
            'names': []
        }

        try:
            # Try using nvidia-smi
            cmd = ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits']
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)

            lines = output.strip().split('\n')
            result['num_gpus'] = len(lines)

            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    result['names'].append(parts[1])
                    result['memory_mb'].append(int(parts[2]))

            self.logger.info(f"Detected {result['num_gpus']} NVIDIA GPU(s)")

        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.info("No NVIDIA GPUs detected or nvidia-smi not available")

        return result


class CircuitAnalyzer:
    """
    Analyzes quantum circuit characteristics relevant for cutting optimization.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_circuit(self, circuit: QuantumCircuit) -> CircuitCharacteristics:
        """
        Analyze circuit and extract relevant characteristics.

        Args:
            circuit: Quantum circuit to analyze

        Returns:
            CircuitCharacteristics with all metrics
        """
        dag = circuit_to_dag(circuit)
        two_qubit_ops = dag.two_qubit_ops()

        # Count gate types
        total_gates = circuit.size()
        two_qubit_gates = len(two_qubit_ops)
        single_qubit_gates = total_gates - two_qubit_gates

        # Count specific gate types
        cnot_gates = sum(
            1 for op in two_qubit_ops
            if op.op.name in ['cx', 'cnot']
        )

        return CircuitCharacteristics(
            num_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            total_gates=total_gates,
            two_qubit_gates=two_qubit_gates,
            cnot_gates=cnot_gates,
            single_qubit_gates=single_qubit_gates,
            circuit=circuit  # Store the actual circuit object
        )
