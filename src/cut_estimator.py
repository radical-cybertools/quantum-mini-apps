"""
cut_estimator.py

Heuristic circuit-cutting estimator using Qiskit circuits + backends.

Usage example
-------------
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from cut_estimator import CutEstimator

qc = QuantumCircuit(34)
# ... build your circuit ...

backend = Aer.get_backend("aer_simulator")

estimator = CutEstimator(
    backend=backend,
    parallelism=16,
    candidate_cuts=(2, 4, 6, 8, 16),
    shots=20_000,
    w_time=1.0,
    w_fidelity=0.1,
)

best_cuts, info = estimator.estimate_best_cuts(qc)
print("Chosen number of cuts:", best_cuts)
for c, stats in sorted(info.items()):
    print(
        f"cuts={c}: "
        f"subexp={stats['num_sub_experiments']:.2e}, "
        f"overhead={stats['sampling_overhead']:.2e}, "
        f"time≈{stats['total_time_sec']:.3f}s, "
        f"cost={stats['cost']:.2f}"
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, Any

import numpy as np
from qiskit import QuantumCircuit




# ============================================================================
# TUNABLE PARAMETERS - Adjust these to calibrate the model
# ============================================================================
#
# To tune the model, you have two options:
#
# 1. Modify DEFAULT_MODEL_CONFIG below (affects all new CutEstimator instances)
# 2. Pass a custom ModelConfig to CutEstimator.__init__(model_config=...)
#
# Example: Make model 10% more conservative
#   custom_config = ModelConfig(c_base=3.2e-13, safety_margin=1.1)
#   estimator = CutEstimator(..., model_config=custom_config)
#
# Key parameters:
#   - c_base: Base time constant (increase for slower predictions)
#   - safety_margin: Final multiplier (use >1.0 for conservative estimates)
#   - gate_density_coeff: Gate density correction (increase for gate-heavy circuits)
#   - overhead_*: Overhead time components (adjust based on actual measurements)
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for time estimation model parameters.
    
    All parameters can be tuned here without modifying the rest of the code.
    """
    # Per-shot time estimation (simulator)
    alpha: float = 2.0  # Exponential scaling factor: 2^(qubits/alpha)
    c_base: float = 2.9e-13  # Base constant calibrated for 30-qubit circuit
    qubit_scale_threshold: int = 30  # Qubit count below which to apply scaling
    qubit_scale_factor: float = 2.2  # Scaling factor: 2^((threshold - qubits) / factor)
    gate_density_coeff: float = 0.035  # Gate density correction coefficient (calibrated for normal-density circuits)
    gate_density_normalize: float = 5.5  # Normalization point for gate density
    min_time_per_shot: float = 3e-10  # Minimum per-shot time (overhead)
    min_time_floor: float = 1e-12  # Absolute minimum return value
    
    # Calibration multiplier to target ratio 0.95-1.05 (actual/predicted)
    # This is applied to per-shot time to adjust overall predictions
    # 
    # IMPORTANT: Actual execution times can vary significantly between runs (2x variation observed)
    # This makes precise calibration challenging. The multiplier should be adjusted based on:
    #   - Your typical execution environment (CPU load, system state, etc.)
    #   - Average actual execution times over multiple runs
    # 
    # Typical values:
    #   - 0.48-0.50: For consistently high actual execution times (~2x baseline)
    #   - 0.70-0.80: For average/mixed execution times
    #   - 0.90-1.00: For consistently low actual execution times
    # 
    # For best results, run test_cut_estimator.py multiple times, average the actual times,
    # and adjust this multiplier to achieve ratios in the 0.95-1.05 range.
    calibration_multiplier: float = 0.70  # Default: middle ground for variable execution times
    
    # Depth estimation
    depth_scale_min: float = 0.1  # Minimum depth scale (10% of original)
    
    # Batch efficiency (for cut circuits)
    batch_efficiency_min: float = 0.4  # Minimum batch efficiency
    batch_efficiency_max: float = 0.7  # Maximum batch efficiency
    batch_efficiency_slope: float = 0.004  # Efficiency improvement per subexperiment
    
    # Overhead time components (for cut circuits)
    overhead_find_cuts: float = 0.7  # Base overhead for finding cuts (seconds)
    overhead_partition_per_subcircuit: float = 0.12  # Partitioning overhead per subcircuit
    overhead_subexp_per_subexperiment: float = 0.04  # Subexperiment generation per subexperiment
    overhead_batch_per_subcircuit: float = 0.12  # Batch setup/teardown per subcircuit
    
    # Gate type detection (for sampling overhead)
    cnot_threshold: float = 0.7  # CNOT gate threshold (70% of two-qubit gates)
    
    # Subexperiment estimation
    subexp_base: float = 10.0  # Base number of subexperiments
    subexp_per_cut: float = 10.0  # Additional subexperiments per cut
    
    # Sampling overhead estimation
    overhead_cnot: float = 9.0  # Overhead factor for CNOT gates
    overhead_iswap: float = 49.0  # Overhead factor for iSwap gates
    overhead_rzz: float = 9.0  # Overhead factor for RZZ-like gates
    overhead_wire: float = 16.0  # Overhead factor for wire cuts


# Global default configuration - modify this to tune the model
DEFAULT_MODEL_CONFIG = ModelConfig()


@dataclass
class HardwareProfile:
    """Approximate performance model for a Qiskit backend.
    
    For real hardware (e.g., IBM Heron 156-qubit devices):
    - max_qubits_per_subcircuit: Maximum qubits that can be reliably executed
    - max_2q_depth: Maximum 2-qubit gate depth for reasonable fidelity
    - Typical constraints for 156-qubit Heron:
      * Qubit count ≤ 156
      * Effective 2Q depth ≤ ~10-20 layers for reasonable results
      * Connectivity should be roughly local to hardware graph
      * Shots ≤ few thousand for "easy" execution
    
    Circuits exceeding these limits should use circuit cutting.
    """
    name: str
    is_simulator: bool
    max_qubits_per_subcircuit: int
    base_shot_time_sec: float  # typical per-layer or per-shot time
    max_2q_depth: int = 50  # Maximum 2-qubit gate depth for reasonable fidelity (hardware only)
    max_shots_easy: int = 5000  # Maximum shots for "easy" execution (hardware only)


class CutEstimator:
    """
    Estimate a reasonable number of cuts for a circuit on a given backend.

    Core API:
        estimator = CutEstimator(backend, parallelism=16, ...)
        best_cuts, debug_info = estimator.estimate_best_cuts(qc)

    debug_info[cuts] contains:
        - num_cuts
        - num_regions
        - subcircuit_qubits
        - qpd_terms_per_cut
        - num_sub_experiments
        - sampling_overhead
        - total_circuit_size
        - hardware_name
        - is_simulator
        - total_time_sec
        - cost
    """

    def __init__(
        self,
        backend: Any,
        parallelism: int,
        candidate_cuts: Iterable[int] = (2, 4, 6, 8, 16),
        shots: int = 10_000,
        num_samples: int = 10,
        w_time: float = 1.0,
        w_fidelity: float = 0.1,
        safety_margin: float = 1.0,
        model_config: ModelConfig | None = None,
        use_estimator: bool = False,
    ) -> None:
        """
        Args:
            backend: Qiskit backend (simulator or real device).
            parallelism: number of subcircuits that can run in parallel.
            candidate_cuts: iterable of cut counts to try.
            shots: shots per sub-experiment.
            num_samples: number of samples for generate_cutting_experiments (limits subexperiments).
            w_time: weight for runtime term in cost (log-scale).
            w_fidelity: weight for "too many cuts" penalty.
            safety_margin: multiplier for conservative time estimates (default 1.0).
                          Use > 1.0 (e.g., 1.1 or 1.2) to ensure predicted >= actual.
            model_config: Model configuration parameters. If None, uses DEFAULT_MODEL_CONFIG.
            use_estimator: If True, accounts for EstimatorV2 overhead (expectation value calculation
                          is slower than backend.run() sampling). Default False.
        """
        self.backend = backend
        self.parallelism = max(int(parallelism), 1)
        self.candidate_cuts = tuple(int(c) for c in candidate_cuts)
        self.shots = int(shots)
        self.num_samples = int(num_samples)
        self.w_time = float(w_time)
        self.w_fidelity = float(w_fidelity)
        self.safety_margin = float(safety_margin)
        self.config = model_config if model_config is not None else DEFAULT_MODEL_CONFIG
        self.use_estimator = bool(use_estimator)  # Whether using EstimatorV2 (slower) vs backend.run()

        self.hw_profile = self._build_hardware_profile_from_backend(backend)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_full_hardware_execution_feasible(
        self,
        quantum_circuit: QuantumCircuit,
        shots: int | None = None,
    ) -> Tuple[bool, str]:
        """
        Determine if a circuit can be feasibly executed on hardware without cutting.
        
        Based on practical constraints for real hardware (e.g., IBM Heron 156-qubit):
        - Qubit count ≤ max_qubits_per_subcircuit
        - Effective 2Q depth ≤ ~10-20 layers for reasonable fidelity
        - Shots ≤ few thousand for "easy" execution
        
        Returns:
            (is_feasible, reason): Tuple indicating feasibility and explanation
        """
        if self.hw_profile.is_simulator:
            # Simulators can handle most circuits
            return True, "Simulator: no hardware constraints"
        
        num_qubits = quantum_circuit.num_qubits
        shots_to_check = shots if shots is not None else self.shots
        
        # Count 2-qubit gates and estimate effective depth
        two_qubit_gates = sum(1 for inst in quantum_circuit.data 
                             if len(inst.qubits) == 2)
        # Rough estimate: effective 2Q depth (gates per qubit)
        avg_2q_depth = two_qubit_gates / max(num_qubits, 1)
        
        reasons = []
        
        # Check qubit count
        if num_qubits > self.hw_profile.max_qubits_per_subcircuit:
            reasons.append(f"qubits ({num_qubits} > {self.hw_profile.max_qubits_per_subcircuit})")
        
        # Check depth
        if avg_2q_depth > self.hw_profile.max_2q_depth:
            reasons.append(f"depth (avg 2Q depth {avg_2q_depth:.1f} > {self.hw_profile.max_2q_depth})")
        
        # Check shots
        if shots_to_check > self.hw_profile.max_shots_easy:
            reasons.append(f"shots ({shots_to_check} > {self.hw_profile.max_shots_easy})")
        
        if reasons:
            reason_str = "Exceeds hardware limits: " + ", ".join(reasons)
            return False, reason_str
        
        return True, f"Feasible: {num_qubits} qubits, ~{avg_2q_depth:.1f} avg 2Q depth, {shots_to_check} shots"

    def predict_memory_requirement(self, num_qubits: int) -> float:
        """
        Predict memory requirement for statevector simulation.
        
        Delegates to model_scale_test.predict_memory_requirement().
        
        Args:
            num_qubits: Number of qubits in the circuit
            
        Returns:
            Required memory in MB
        """
        from model_scale_test import predict_memory_requirement as _predict_memory
        return _predict_memory(num_qubits)
    
    def check_memory_feasibility(
        self, num_qubits: int, max_memory_mb: float | None = None
    ) -> tuple[bool, float, float]:
        """
        Check if a circuit can be executed given memory constraints.
        
        Delegates to model_scale_test.check_memory_feasibility().
        
        Args:
            num_qubits: Number of qubits in the circuit
            max_memory_mb: Maximum available memory in MB. If None, uses hardware profile.
            
        Returns:
            Tuple of (is_feasible, required_memory_mb, max_memory_mb)
        """
        from model_scale_test import check_memory_feasibility as _check_memory
        
        if max_memory_mb is None:
            # Use a default based on hardware profile
            # For simulators, assume 64 GB default
            max_memory_mb = 65536.0 if self.hw_profile.is_simulator else 65536.0
        
        return _check_memory(num_qubits, max_memory_mb)

    def estimate_best_cuts(
        self,
        quantum_circuit: QuantumCircuit,
    ) -> Tuple[int, Dict[int, dict]]:
        """
        Evaluate candidate cut counts and return the best one plus debug info.

        Returns:
            best_num_cuts, debug_info

            debug_info is a dict:
                debug_info[num_cuts] -> feature dict for that configuration.
        """
        num_qubits = quantum_circuit.num_qubits

        total_circuit_size = self._qiskit_total_circuit_size(quantum_circuit)
        qpd_terms_per_cut = self._estimate_qpd_terms_per_cut_from_circuit(
            quantum_circuit
        )

        base_subexp = 1.0
        best_cost = float("inf")
        best_cuts = 0
        debug_info: Dict[int, dict] = {}
        
        # Check if full hardware execution is feasible (for informational purposes)
        if not self.hw_profile.is_simulator:
            is_feasible, reason = self.is_full_hardware_execution_feasible(quantum_circuit)
            # Store feasibility info in debug output
            debug_info['_hardware_feasibility'] = {
                'full_execution_feasible': is_feasible,
                'reason': reason,
            }

        for c in self.candidate_cuts:
            # At most qubits-1 cuts
            if c >= num_qubits:
                continue

            num_regions = c + 1
            subcircuit_qubits = num_qubits / num_regions

            # Hardware constraint: skip if still too large per region
            # Exception: 0 cuts means full circuit execution, so hardware limit doesn't apply
            if c > 0 and subcircuit_qubits > self.hw_profile.max_qubits_per_subcircuit:
                continue

            num_subexp = self._estimate_subexperiments(c, qpd_terms_per_cut)
            sampling_ovh = self._estimate_sampling_overhead(c, quantum_circuit)
            total_time = self._estimate_total_time(
                circ=quantum_circuit,
                num_subexperiments=num_subexp,
                subcircuit_qubits=subcircuit_qubits,
                sampling_overhead=sampling_ovh,
                num_cuts=c,
            )

            # Tradeoff analysis:
            # More cuts → smaller subcircuits → exponentially faster per-shot (2^qubits scaling)
            # BUT also → more subexperiments (6^cuts) × higher overhead (9^cuts)
            # The exponential growth in (subexperiments × overhead) often overwhelms
            # the per-shot time savings, making fewer cuts optimal.
            # This is why 2 cuts often wins: good balance between subcircuit size reduction
            # and overhead growth.

            # Simple fidelity penalty: more cuts → worse effective fidelity
            fidelity_penalty = c

            # log-scale time normalization
            time_norm = np.log10(max(total_time, 1e-12))

            cost = self.w_time * time_norm + self.w_fidelity * fidelity_penalty

            stats = {
                "num_cuts": c,
                "num_regions": num_regions,
                "subcircuit_qubits": subcircuit_qubits,
                "qpd_terms_per_cut": qpd_terms_per_cut,
                "num_sub_experiments": num_subexp,
                "sampling_overhead": sampling_ovh,
                "total_circuit_size": total_circuit_size,
                "hardware_name": self.hw_profile.name,
                "is_simulator": self.hw_profile.is_simulator,
                "total_time_sec": total_time,
                "cost": float(cost),
            }
            debug_info[c] = stats

            if cost < best_cost:
                best_cost = cost
                best_cuts = c

        return best_cuts, debug_info

    # ------------------------------------------------------------------
    # Internal helpers: hardware profiling
    # ------------------------------------------------------------------

    def _build_hardware_profile_from_backend(self, backend: Any) -> HardwareProfile:
        cfg = backend.configuration()

        is_sim = bool(getattr(cfg, "simulator", False))
        name = getattr(cfg, "backend_name", repr(cfg))
        max_qubits = int(getattr(cfg, "n_qubits", 32))

        gate_length = None
        try:
            props = backend.properties()
        except Exception:
            props = None

        if props is not None:
            # Try some common 2-qubit gates; use first successful gate_length
            for gname in ("cx", "cz", "ecr", "iswap"):
                try:
                    # Use qubits [0, 1] as a heuristic; gate_length
                    # will raise if unsupported.
                    gate_length = float(props.gate_length(gname, [0, 1]))
                    break
                except Exception:
                    continue

        # Fallbacks
        if gate_length is None:
            gate_length = 1e-6

        if is_sim:
            # For simulators, base_shot_time_sec is just a floor; we
            # use a separate scaling law on top of it.
            # Note: Actual AerSimulator is much faster than this suggests,
            # but this is used as a baseline for the exponential scaling model.
            return HardwareProfile(
                name=name,
                is_simulator=True,
                max_qubits_per_subcircuit=max_qubits,
                base_shot_time_sec=1e-7,
                max_2q_depth=1000,  # Simulators can handle deep circuits
                max_shots_easy=100000,  # Simulators can handle many shots
            )

        # Physical backend: treat gate_length as typical per-layer time scale
        # For real hardware (e.g., IBM Heron 156-qubit), set realistic constraints
        # Based on empirical data: shallow circuits (depth ≤ 20) work well,
        # deeper circuits (depth > 50) become dominated by noise
        return HardwareProfile(
            name=name,
            is_simulator=False,
            max_qubits_per_subcircuit=max_qubits,
            base_shot_time_sec=gate_length,
            max_2q_depth=20,  # Conservative: 10-20 layers for reasonable fidelity
            max_shots_easy=4000,  # Few thousand shots for "easy" execution
        )

    # ------------------------------------------------------------------
    # Internal helpers: circuit statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _qiskit_total_circuit_size(circ: QuantumCircuit) -> int:
        """Size ≈ total gate count."""
        return len(circ.data)

    @staticmethod
    def _qiskit_two_qubit_gate_count(circ: QuantumCircuit) -> int:
        """Count common 2-qubit gates as a proxy for entangling structure."""
        ops = circ.count_ops()
        twoq_names = ["cx", "cz", "ecr", "iswap", "rxx", "ryy", "rzz", "rzx"]
        return int(sum(ops.get(name, 0) for name in twoq_names))

    def _estimate_qpd_terms_per_cut_from_circuit(
        self,
        circ: QuantumCircuit,
    ) -> float:
        """
        Heuristic QPD expansion terms per cut (number of subexperiments per cut).

        Based on qiskit-addon-cutting documentation:
        - Gate cuts (e.g., RZZGate): 6 subexperiments per cut
        - Wire cuts: 4 subexperiments per cut (LO setting)
        
        For gate cuts, the total number of subexperiments scales as 6^num_cuts.
        This matches the implementation in circuit_cutting_dreamer/motif.py.
        
        Reference: https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html#circuit-cutting-as-a-quasiprobability-decomposition-qpd
        """
        size = self._qiskit_total_circuit_size(circ)
        twoq = self._qiskit_two_qubit_gate_count(circ)

        if size == 0:
            return 1.0

        # Base value: 6 for gate cuts (as per RZZGate example in documentation)
        # Wire cuts would use 4, but gate cuts are more common in practice
        base_terms = 6.0
        
        # Optionally scale slightly based on 2-qubit gate density
        # (higher density may indicate more complex gate cuts)
        twoq_density = twoq / max(size, 1)
        return base_terms * (1.0 + 0.1 * twoq_density)

    def _estimate_overhead_factor_per_cut_from_circuit(
        self,
        circ: QuantumCircuit,
    ) -> float:
        """
        Estimate sampling overhead factor per cut.
        
        According to qiskit-addon-cutting documentation:
        - Sampling overhead = (sum |a_i|)^2 where a_i are QPD coefficients
        - For multiple cuts, total overhead = product of individual overhead factors
        - CNOT gate: overhead factor = 9 (3^2)
        - iSwapGate: overhead factor = 49 (7^2)
        - RZZGate: overhead factor = [1 + 2|sin(θ)|]^2 (varies with theta, typically ~6-36)
        - Wire cuts: overhead factor = 16 (4^2) per cut (LO setting)
        
        For gate cuts with 6 subexperiments (RZZGate-like), we estimate overhead factor
        based on typical coefficient distributions. A conservative estimate is to use
        the square of the number of subexperiments as an upper bound, but in practice
        the overhead is typically lower.
        
        Reference: https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html#sampling-overhead-reference-table
        """
        size = self._qiskit_total_circuit_size(circ)
        twoq = self._qiskit_two_qubit_gate_count(circ)
        ops = circ.count_ops()

        if size == 0:
            return 1.0

        # Check for specific gate types that have known overhead factors
        # CNOT gates: overhead = 9
        cx_count = ops.get('cx', 0)
        # iSwapGate: overhead = 49
        iswap_count = ops.get('iswap', 0)
        
        # If circuit is dominated by CNOT gates, use CNOT overhead factor
        if cx_count > twoq * self.config.cnot_threshold:
            return self.config.overhead_cnot
        # If circuit has iSwap gates, use higher overhead
        elif iswap_count > 0:
            return self.config.overhead_iswap
        
        # For RZZGate-like circuits (6 subexperiments), estimate overhead factor
        # Based on documentation: RZZGate overhead = [1 + 2|sin(θ)|]^2
        # For typical theta values, this ranges from ~1 to ~36
        # We use a conservative estimate: if we have 6 subexperiments, assume
        # coefficients sum to approximately sqrt(6) ≈ 2.45, so overhead ≈ 6
        # But for more accurate estimation, we can use a value between 6-36
        # For typical variational circuits with RZZ gates, overhead is often ~9-16
        
        # Base overhead factor for gate cuts (conservative estimate for RZZGate-like)
        # Using RZZ overhead as a middle ground for typical variational circuits
        base_overhead = self.config.overhead_rzz
        
        # Adjust based on 2-qubit gate density
        # Higher density might indicate more complex cuts with higher overhead
        twoq_density = twoq / max(size, 1)
        if twoq_density > 0.5:  # High density
            return base_overhead * 1.2  # ~10.8
        else:
            return base_overhead  # 9.0

    # ------------------------------------------------------------------
    # Internal helpers: subexperiments, overhead, timing
    # ------------------------------------------------------------------

    def _estimate_subexperiments(self, num_cuts: int, qpd_terms_per_cut: float) -> float:
        """
        Estimate number of subexperiments generated by generate_cutting_experiments.
        
        According to qiskit-addon-cutting:
        - The theoretical maximum is qpd_terms_per_cut^num_cuts (e.g., 6^num_cuts for RZZ-like gates)
        - The num_samples parameter limits the number of samples drawn from the 
          quasi-probability distribution, which in turn limits the number of subexperiments
        - The actual number of subexperiments is typically: num_samples * num_subcircuits
          but cannot exceed the theoretical maximum
        
        Reference: qiskit-addon-cutting documentation and motif.py line 465:
        actual_num_samples = num_sub_circuits * (6 ** len(metadata["cuts"]))
        This suggests the theoretical max per subcircuit is 6^num_cuts, but num_samples
        limits how many are actually generated.
        """
        if num_cuts <= 0:
            return 1.0
        
        # Theoretical maximum: exponential in number of cuts
        # For each cut, we get qpd_terms_per_cut subexperiments
        # Total: qpd_terms_per_cut^num_cuts
        theoretical_max = float(qpd_terms_per_cut ** num_cuts)
        
        # Number of subcircuits created by cutting
        num_subcircuits = num_cuts + 1
        
        # The num_samples parameter limits how many samples are drawn per subcircuit
        # Based on qiskit-addon-cutting behavior: num_samples limits the number of
        # subexperiments generated per subcircuit, so total ≈ num_samples * num_subcircuits
        # However, this cannot exceed the theoretical maximum
        estimated_from_samples = self.num_samples * num_subcircuits
        
        # The actual number is the minimum of:
        # 1. Theoretical maximum (exponential growth)
        # 2. Sample-limited estimate (num_samples * num_subcircuits)
        # This matches the behavior where num_samples acts as a cap on subexperiments
        return min(theoretical_max, estimated_from_samples)

    def _estimate_sampling_overhead(
        self,
        num_cuts: int,
        circ: QuantumCircuit,
    ) -> float:
        """
        Estimate sampling overhead based on number of cuts and circuit structure.
        
        According to qiskit-addon-cutting documentation:
        - Sampling overhead = product of overhead factors for each cut
        - Each cut type has a specific overhead factor (e.g., CNOT: 9, iSwapGate: 49)
        - For multiple cuts, multiply the overhead factors
        
        Reference: https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html#sampling-overhead-reference-table
        
        Args:
            num_cuts: Number of cuts in the circuit
            circ: Quantum circuit to analyze
            
        Returns:
            Estimated sampling overhead factor
        """
        if num_cuts <= 0:
            return 1.0
        
        # Get overhead factor per cut based on circuit structure
        overhead_per_cut = self._estimate_overhead_factor_per_cut_from_circuit(circ)
        
        # Total overhead = product of overhead factors for all cuts
        # This matches the documentation: "multiply the overhead factors for all cuts"
        return float(overhead_per_cut ** num_cuts)

    def _estimate_time_per_shot(
        self,
        circ: QuantumCircuit,
        subcircuit_qubits: float,
        use_estimator: bool = False,
    ) -> float:
        """
        Rough per-shot time model for subcircuits.

        - Physical backend:
            per-shot time ≈ subcircuit_depth * base_shot_time_sec
            (subcircuit_depth is roughly proportional to original depth, 
             but may be reduced due to cutting)

        - Simulator:
            per-shot time ≈ c * subcircuit_depth * 2^(subcircuit_qubits / alpha)
            The exponential factor in qubits is the KEY benefit of cutting:
            smaller subcircuits = exponentially faster per-shot execution

        Both c and alpha are tunable constants.
        
        Note: The subcircuit size (qubits) has a major impact on simulator performance
        due to the exponential scaling. This is why cutting can be beneficial despite
        the overhead in subexperiments and sampling.
        """
        # Estimate subcircuit depth (roughly proportional to original, but may be reduced)
        # In practice, cutting may reduce depth, but we use a conservative estimate
        full_depth = max(int(circ.depth()), 1)
        # Subcircuit depth is roughly proportional to subcircuit size
        # This is a heuristic - actual depth depends on how cuts are made
        num_qubits = circ.num_qubits
        depth_scale = max(subcircuit_qubits / num_qubits, self.config.depth_scale_min)
        subcircuit_depth = max(int(full_depth * depth_scale), 1)

        if not self.hw_profile.is_simulator:
            # Physical device: subcircuit_depth × typical layer time
            # Smaller subcircuits = fewer gates = faster execution
            return subcircuit_depth * self.hw_profile.base_shot_time_sec

        # Simulator: exponential in qubits - THIS IS THE KEY BENEFIT OF CUTTING
        # Smaller subcircuits = exponentially faster per-shot execution
        # Example: 30 qubits → 10 qubits: 2^(30/2) / 2^(10/2) = 2^15 / 2^5 = 32768 / 32 = 1024x faster!
        # 
        # Calibration based on AerSimulator measurements:
        # For 30-qubit EfficientSU2 circuit (depth=37, gates=238), 20k shots: actual ~0.007s
        # Per-shot time ≈ 0.007 / 20000 = 3.5e-7s
        # For 20-qubit EfficientSU2 circuit (depth=27, gates=158), 20k shots: actual ~0.004s
        # Per-shot time ≈ 0.004 / 20000 = 2.0e-7s
        # 
        # Model: c * depth * gate_density_factor * 2^(qubits/alpha) + min_time
        # where gate_density_factor = 1 + 0.1 * (gates/depth - 5.0)
        # This accounts for both depth (primary) and gate density (correction)
        # 
        # For 30-qubit: depth=37, gates=238, density=6.43
        #   c * 37 * (1 + 0.1*(6.43-5.0)) * 32768 = 3.5e-7
        #   c * 37 * 1.143 * 32768 = 3.5e-7
        #   c ≈ 2.52e-13
        # For 20-qubit: depth=27, gates=158, density=5.85
        #   c * 27 * (1 + 0.1*(5.85-5.0)) * 1024 = 2.0e-7
        #   c * 27 * 1.085 * 1024 = 2.0e-7
        #   c ≈ 6.66e-12
        # 
        # The constants still differ significantly. Let's use a simpler approach:
        # Use depth as primary, but scale constant based on qubit count to account for
        # different circuit characteristics at different sizes.
        alpha = self.config.alpha
        
        # Base constant calibrated for 30-qubit circuit
        c_base = self.config.c_base
        
        # Scale constant based on qubit count to account for different circuit characteristics
        # For smaller circuits, the constant needs to be significantly larger
        # For larger circuits (>30 qubits), AerSimulator may use optimizations/approximations
        # that reduce the exponential scaling, so we need to reduce the constant aggressively
        if subcircuit_qubits < self.config.qubit_scale_threshold:
            # Smaller circuits: increase constant
            qubit_scale = 2.0 ** ((self.config.qubit_scale_threshold - subcircuit_qubits) / self.config.qubit_scale_factor)
        elif subcircuit_qubits > 30:
            # Larger circuits: reduce constant to account for simulator optimizations
            # Empirical data shows actual per-shot time is relatively constant (~2-3e-7s)
            # regardless of qubit count, suggesting optimizations break exponential scaling
            # We need aggressive scaling: for 49 qubits, need ~1/1000 of base constant
            excess_qubits = subcircuit_qubits - 30
            # Reduce constant exponentially: 2^(-excess_qubits / 2.5) provides aggressive scaling
            # This compensates for the exponential growth in 2^(qubits/2)
            qubit_scale = 2.0 ** (-excess_qubits / 2.5)
        else:
            qubit_scale = 1.0
        c = c_base * qubit_scale
        
        # Get gate density for correction factor
        num_qubits = circ.num_qubits
        full_gates = len(circ.data)
        full_depth = max(int(circ.depth()), 1)
        gate_density = full_gates / full_depth if full_depth > 0 else self.config.gate_density_normalize
        
        # Check for RXX/RYY gates which are slower than CNOT gates
        # Note: RZZ gates are handled separately and don't need this correction
        has_rxx_ryy = False
        for instruction in circ.data:
            if instruction.operation.name in ['rxx', 'ryy']:
                has_rxx_ryy = True
                break
        
        # Gate density correction: circuits with more gates per layer take longer
        # Normalize around density ~5-6 (typical for EfficientSU2)
        # For very high density circuits (>8), use stronger correction
        if gate_density > 8.0:
            # High density circuits need stronger correction
            excess_density = gate_density - self.config.gate_density_normalize
            # Use increased coefficient for high density circuits
            high_density_coeff = self.config.gate_density_coeff * 2.5
            gate_density_factor = 1.0 + high_density_coeff * excess_density
        else:
            # Normal density: linear correction
            gate_density_factor = 1.0 + self.config.gate_density_coeff * (gate_density - self.config.gate_density_normalize)
        
        # Additional correction for RXX/RYY gates which are slower
        if has_rxx_ryy:
            gate_density_factor *= 1.6  # RXX/RYY gates are ~1.6x slower than CNOT
        
        # Calculate base per-shot time using depth with gate density correction
        base_time = c * subcircuit_depth * gate_density_factor * (2 ** (subcircuit_qubits / alpha))
        
        # For very large circuits (>35 qubits), AerSimulator may use optimizations
        # that make execution time relatively constant rather than exponential
        # Empirical data shows per-shot time varies significantly (~3.5-9e-7s) for large circuits
        # Add a cap to prevent overestimation while keeping ratio between 0.8-1.0
        if subcircuit_qubits > 35:
            # Cap the exponential growth - use a maximum per-shot time
            # Based on empirical data: actual per-shot time varies (3.5-9e-7s) for large circuits
            # Adjusted to target ratio 0.8-1.0 (predicted >= actual, but not too conservative)
            # For 49 qubits, actual varies ~0.007-0.018s for 20k shots = 3.5-9e-7s per-shot
            # Use a cap that accounts for variability - target upper end of range
            # Cap based on observed execution times: actual varies ~0.007-0.018s
            # For 0.018s actual, to get ratio=0.9, need predicted=0.020s = 1.0e-6s per-shot
            # Use slightly lower to account for average case
            max_per_shot_time = 9.5e-7  # Cap to achieve ratio ~0.8-1.0 accounting for execution time variability
            base_time = min(base_time, max_per_shot_time)
        
        # Add overhead for very fast circuits (initialization, etc.)
        # For very shallow circuits (depth < 5), add extra overhead as they have
        # relatively more overhead compared to execution time
        min_time = self.config.min_time_per_shot
        if full_depth < 5:
            # Shallow circuits have proportionally more overhead
            # Scale min_time based on how shallow the circuit is
            depth_factor = max(1.0, 5.0 / max(full_depth, 1))
            min_time = self.config.min_time_per_shot * depth_factor
        
        per_shot = base_time + min_time
        
        # Apply calibration multiplier to target ratio 0.95-1.05 (actual/predicted)
        # This adjusts overall predictions to match actual execution times
        multiplier = self.config.calibration_multiplier
        
        # If using EstimatorV2 (expectation value calculation), apply additional overhead
        # EstimatorV2 is much slower than backend.run() because it computes expectation values
        # Overhead scales with circuit size based on observed ratios:
        #   - 12 qubits: ~1.8-2x overhead
        #   - 14 qubits: ~3.5-4x overhead
        #   - 16 qubits: ~5-6x overhead
        #   - 18 qubits: ~7-8x overhead
        #   - 20 qubits: ~12-13x overhead
        #   - 30 qubits: ~6000x overhead (extreme, likely due to MPS approximation method)
        # 
        # NOTE: For circuits >25 qubits, AerSimulator may switch to MPS approximation,
        # which has fundamentally different performance characteristics. The overhead model
        # may not accurately capture this for very large circuits. Consider using circuit
        # cutting for circuits >25 qubits to get more predictable performance.
        # Use piecewise scaling: linear for small-medium, exponential for large
        if use_estimator:
            if subcircuit_qubits < 15:
                # Small circuits: linear scaling ~2-4x
                estimator_overhead = 1.5 + 0.2 * (subcircuit_qubits - 12)
            elif subcircuit_qubits < 25:
                # Medium circuits: linear scaling ~4-20x
                estimator_overhead = 4.0 + 1.6 * (subcircuit_qubits - 15)
            else:
                # Large circuits: extreme overhead due to approximation method switching
                # For 30q, need ~5600x overhead to match observed ratios
                # The 30q circuit likely uses MPS approximation which is much slower for EstimatorV2
                # Use very aggressive exponential scaling
                # For 30q: overhead * 2^((30-25)/scale) = 5600
                # If scale=0.5: overhead * 2^10 = overhead * 1024 = 5600, so overhead = 5.5
                # But we want smoother scaling, so use: base * 2^((qubits-25)/0.5)
                # For 30q: base * 2^10 = base * 1024 = 5600, so base ≈ 5.5
                # Use base=6.0 for safety margin
                estimator_overhead = 6.0 * (2.0 ** ((subcircuit_qubits - 25) / 0.5))
            multiplier = multiplier * estimator_overhead
        
        per_shot = per_shot * multiplier
        
        # Ensure we don't return zero or negative values
        return max(per_shot, self.config.min_time_floor)

    def _estimate_total_time(
        self,
        circ: QuantumCircuit,
        num_subexperiments: float,
        subcircuit_qubits: float,
        sampling_overhead: float = 1.0,
        num_cuts: int = 0,
    ) -> float:
        """
        Estimate total execution time for circuit cutting.
        
        According to qiskit-addon-cutting documentation:
        - Sampling overhead is "the factor by which the overall number of shots must increase"
        - This means: effective_shots = base_shots * sampling_overhead
        - Total time = overhead_time + (num_subexperiments * effective_shots * per_shot_time) / parallelism
        
        Reference: https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html#circuit-cutting-as-a-quasiprobability-decomposition-qpd
        
        Args:
            circ: Quantum circuit
            num_subexperiments: Number of subexperiments to execute
            subcircuit_qubits: Number of qubits per subcircuit
            sampling_overhead: Sampling overhead factor (multiplies required shots)
            num_cuts: Number of cuts (for overhead estimation)
            
        Returns:
            Estimated total execution time in seconds
        """
        # Use the estimator flag from instance
        per_shot_time = self._estimate_time_per_shot(circ, subcircuit_qubits, use_estimator=self.use_estimator)
        
        # CRITICAL: Sampling overhead is NOT applied to shots per subexperiment!
        # Looking at actual execution: sampler.run(subexp_list, shots=shots) uses base shots
        # The sampling overhead is already accounted for in the NUMBER of subexperiments,
        # not in shots per subexperiment. More cuts → more subexperiments needed (already in num_subexperiments)
        # So we use base shots, not effective_shots
        base_shots = self.shots
        
        # Execution time for running subexperiments
        # CRITICAL INSIGHT: Based on actual execution pattern in test_cut_estimator.py:
        # ```python
        # with Batch(backend=backend) as batch:
        #     sampler = SamplerV2(mode=batch)
        #     for label, subexp_list in subexperiments.items():  # SEQUENTIAL LOOP!
        #         job = sampler.run(subexp_list, shots=shots)  # Uses base shots, not multiplied!
        #         result = job.result()
        # ```
        # 
        # Key points:
        # 1. Subcircuits (labels) are executed SEQUENTIALLY - one at a time
        # 2. Within each subcircuit, subexperiments run in Batch mode (may have some parallelism)
        # 3. More cuts = more subcircuits = more sequential overhead
        # 4. BUT: more cuts = smaller subcircuits = faster per-shot time
        # 5. Sampling overhead is in num_subexperiments, NOT in shots per subexperiment
        #
        # So: execution_time = sum over subcircuits(time_per_subcircuit)
        #     where time_per_subcircuit depends on subexperiments in that subcircuit
        
        if num_subexperiments <= 1.0 and sampling_overhead <= 1.0:
            # Full circuit execution: no parallelism benefit
            execution_time = num_subexperiments * base_shots * per_shot_time
        else:
            # For cut circuits: subcircuits run SEQUENTIALLY
            num_subcircuits = num_cuts + 1
            
            # Estimate subexperiments per subcircuit
            # Based on actual data: subexperiments are roughly evenly distributed
            subexp_per_subcircuit = num_subexperiments / num_subcircuits
            
            # Time per subcircuit: depends on batch execution of subexperiments
            # Batch mode provides some parallelism, but subexperiments still take time
            # Based on actual data, batch efficiency seems to be around 40-70%
            # Efficiency improves with more subexperiments per subcircuit (better batching)
            batch_efficiency = min(
                self.config.batch_efficiency_max,
                self.config.batch_efficiency_min + self.config.batch_efficiency_slope * subexp_per_subcircuit
            )
            
            # Time to execute one subcircuit's batch of subexperiments
            # Use base_shots (not multiplied by overhead - overhead is in num_subexperiments)
            time_per_subcircuit = (subexp_per_subcircuit * base_shots * per_shot_time) * batch_efficiency
            
            # Total execution time: sequential sum over all subcircuits
            # This is why more cuts = longer time (more sequential steps)
            # BUT smaller subcircuits = faster per-shot helps offset this
            execution_time = num_subcircuits * time_per_subcircuit
        
        # Add overhead time for cutting operations:
        # Based on actual measurements (reverse engineering from actual vs predicted):
        # - 2 cuts: actual 6.877s, predicted exec ~3.4s → overhead ~3.5s
        # - 4 cuts: actual 11.162s, predicted exec ~6.3s → overhead ~4.9s
        # - 6 cuts: actual 12.970s, predicted exec ~13.0s → overhead ~0s (execution dominates)
        #
        # Overhead components:
        # - Finding cuts: ~0.5-1.0s (constant)
        # - Partitioning: ~0.1-0.2s per subcircuit
        # - Generating subexperiments: ~0.03-0.05s per subexperiment
        # - Batch setup/teardown: ~0.1-0.15s per subcircuit
        overhead_time = 0.0
        if num_cuts > 0:
            num_subcircuits = num_cuts + 1
            
            # Base overhead: finding cuts (relatively constant)
            overhead_time = self.config.overhead_find_cuts
            
            # Partitioning overhead: scales with number of subcircuits
            overhead_time += self.config.overhead_partition_per_subcircuit * num_subcircuits
            
            # Subexperiment generation: scales with total subexperiments
            overhead_time += self.config.overhead_subexp_per_subexperiment * num_subexperiments
            
            # Batch setup/teardown: per subcircuit in sequential execution
            overhead_time += self.config.overhead_batch_per_subcircuit * num_subcircuits
        
        total_time = execution_time + overhead_time
        
        # Apply safety margin for conservative estimates (predicted >= actual)
        # This ensures the model doesn't underestimate execution time
        return total_time * self.safety_margin
