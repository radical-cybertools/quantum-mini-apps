"""
Mock AerSimulator backend that uses execution time and memory models
instead of actually executing circuits.

This is useful for:
- Q-Dreamer optimization/planning without actual execution
- Testing circuit cutting strategies
- Predicting execution feasibility before running
- Fast iteration on optimization algorithms
"""

import time
import numpy as np
from typing import Any, Optional, Union
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import PrimitiveResult
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.jobs import AerJob
from model_scale_test import (
    predict_execution_time,
    predict_memory_requirement,
    check_memory_feasibility
)


class MockAerSimulator(AerSimulator):
    """
    Mock AerSimulator that extends AerSimulator and overrides run() method
    to use execution time and memory models instead of actually executing circuits.
    
    ✅ Why extend AerSimulator instead of reimplementing?
    - Inherits all AerSimulator functionality and compatibility automatically
    - Works with any code expecting AerSimulator (type checking, isinstance, etc.)
    - No need to reimplement interface methods (set_options, configuration, etc.)
    - Future-proof: automatically inherits updates to AerSimulator
    - Cleaner, more maintainable code
    
    This backend:
    - Inherits all AerSimulator functionality and compatibility
    - Overrides run() to use models instead of actual execution
    - Predicts execution time using the calibrated model
    - Checks memory feasibility (replicates AerSimulator's MemoryError behavior)
    - Returns mock results (can be configured)
    - Simulates execution delay based on predicted time
    
    Note on memory checking:
    - Real AerSimulator checks memory DURING execution (in job.result())
    - Since we don't actually execute, we check in run() to replicate the behavior
    - This allows fast failure prediction for Q-Dreamer optimization
    - Uses the same memory formula as AerSimulator: (2^N * 16 bytes) / (1024^2)
    
    Usage:
        from mock_aer_simulator import MockAerSimulator
        
        # Use like regular AerSimulator
        backend = MockAerSimulator(max_memory_mb=65536, method="statevector")
        
        # Works with backend.run() directly
        job = backend.run(circuit)
        result = job.result()
        
        # Works with EstimatorV2 (may need additional compatibility for full features)
        from qiskit_aer.primitives import EstimatorV2
        estimator = EstimatorV2.from_backend(backend)
    """
    
    def __init__(
        self,
        max_memory_mb: float = 65536.0,  # 64 GB default
        simulate_delay: bool = False,
        mock_expectation_value: Optional[float] = None,
        **aer_simulator_kwargs
    ):
        """
        Initialize mock AerSimulator.
        
        Args:
            max_memory_mb: Maximum available memory in MB
            simulate_delay: If True, sleep for predicted execution time
            mock_expectation_value: If provided, return this value instead of computing
            **aer_simulator_kwargs: Additional arguments passed to AerSimulator.__init__
                                   (e.g., method="statevector", device="CPU")
        """
        # Initialize parent AerSimulator with provided kwargs
        # Default to statevector method if not specified
        if "method" not in aer_simulator_kwargs:
            aer_simulator_kwargs["method"] = "statevector"
        super().__init__(**aer_simulator_kwargs)
        
        # Store mock-specific attributes
        self._max_memory_mb = max_memory_mb
        self._simulate_delay = simulate_delay
        self._mock_expectation_value = mock_expectation_value
        
    def run(
        self, 
        circuits: Union[QuantumCircuit, list[QuantumCircuit]], 
        parameter_binds: Optional[list] = None,
        **run_options
    ):
        """
        Override run() to use models instead of actual execution.
        
        This method:
        1. Checks memory feasibility
        2. Predicts execution time
        3. Optionally simulates delay
        4. Returns mock results compatible with AerSimulator
        
        Args:
            circuits: Single circuit or list of circuits
            parameter_binds: Parameter bindings (not used in mock)
            **run_options: Additional run options
            
        Returns:
            MockAerJob that behaves like AerJob but uses models
        """
        # Handle single circuit or list
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        
        # Note: Real AerSimulator checks memory DURING execution (in job.result())
        # We'll check in MockAerJob.result() to match that behavior exactly
        # This way we replicate AerSimulator's timing: run() succeeds, result() may fail
        
        # Create mock job that will simulate execution
        return MockAerJob(circuits, self, parameter_binds, run_options)


class MockAerJob(AerJob):
    """
    Mock AerJob that extends AerJob and overrides result() to use models.
    
    This provides full compatibility with AerSimulator's job interface
    while using models instead of actual execution.
    """
    
    def __init__(
        self, 
        circuits: list[QuantumCircuit], 
        backend: MockAerSimulator,
        parameter_binds: Optional[list],
        run_options: dict
    ):
        """
        Initialize mock job.
        
        Args:
            circuits: List of circuits to "execute"
            backend: MockAerSimulator backend
            parameter_binds: Parameter bindings (not used)
            run_options: Run options (not used)
        """
        # Store backend and circuits before creating mock function
        self._mock_backend = backend
        self._circuits = circuits
        self._parameter_binds = parameter_binds
        self._run_options = run_options
        self._mock_result = None
        
        # Initialize parent - AerJob.__init__(backend, job_id, fn, circuits, parameter_binds, run_options, executor)
        # fn is the execution function - we'll provide a mock one
        import uuid
        job_id = str(uuid.uuid4())
        
        # Create a mock execution function that uses models
        def mock_exec_fn(circuits, parameter_binds, run_options):
            """Mock execution function that uses models."""
            return self._simulate_execution()
        
        super().__init__(
            backend=backend,
            job_id=job_id,
            fn=mock_exec_fn,
            circuits=circuits,
            parameter_binds=parameter_binds,
            run_options=run_options,
            executor=None
        )
        
    def result(self):
        """
        Override result() to return mock results using models.
        
        Returns:
            Result object compatible with AerSimulator results
        """
        if self._mock_result is None:
            self._mock_result = self._simulate_execution()
        return self._mock_result
    
    def _simulate_execution(self):
        """Simulate circuit execution using models."""
        from qiskit.result import Result
        
        # Process each circuit
        results = []
        for idx, circuit in enumerate(self._circuits):
            num_qubits = circuit.num_qubits
            depth = circuit.depth()
            
            # Check memory feasibility (replicates AerSimulator's check during execution)
            # Real AerSimulator checks memory in job.result(), so we check here too
            is_feasible, required_mb, max_mb = check_memory_feasibility(
                num_qubits, self._mock_backend._max_memory_mb
            )
            
            if not is_feasible:
                # Replicate AerSimulator's MemoryError with same message format
                error_msg = (
                    f"ERROR: Insufficient memory to run circuit using the statevector simulator. "
                    f"Required memory: {required_mb:.0f}M, max memory: {max_mb:.0f}M"
                )
                raise MemoryError(error_msg)
            
            # Predict execution time
            predicted_time = predict_execution_time(
                num_qubits=num_qubits,
                depth=depth,
                reps=2  # Default, could be extracted from circuit
            )
            
            # Simulate delay if requested
            if self._mock_backend._simulate_delay:
                time.sleep(min(predicted_time, 1.0))  # Cap at 1 second for testing
            
            # Generate mock expectation value
            if self._mock_backend._mock_expectation_value is not None:
                expval = self._mock_backend._mock_expectation_value
            else:
                # Use deterministic but circuit-dependent value
                expval = self._compute_mock_expectation_value(circuit)
            
            # Get shots from run options
            shots = self._run_options.get("shots", 1024)
            
            # Calculate required memory
            required_mb = predict_memory_requirement(num_qubits)
            
            # Determine number of classical bits (from measurements)
            num_clbits = circuit.num_clbits if circuit.num_clbits > 0 else num_qubits
            
            # Create header matching AerSimulator format
            header = {
                "name": f"circuit-{idx}",
                "n_qubits": num_qubits,
                "qreg_sizes": [["q", num_qubits]],
                "creg_sizes": [["meas", num_clbits]] if num_clbits > 0 else [],
                "memory_slots": num_clbits,
                "global_phase": 0.0,
                "metadata": {}
            }
            
            # Create metadata matching AerSimulator format
            # These fields are what AerSimulator actually includes
            metadata = {
                # Execution info
                "time_taken": predicted_time,
                "num_bind_params": 1,
                "runtime_parameter_bind": False,
                
                # Parallelism info
                "parallel_state_update": 16,  # Typical default
                "parallel_shots": 1,
                
                # Measurement info
                # sample_measure_time: Typically a small fraction of total time
                # For small circuits, it's usually very small (< 1ms)
                "sample_measure_time": min(predicted_time * 0.1, 0.001) if predicted_time > 0 else 0.0001,
                "measure_sampling": True,
                "num_clbits": num_clbits,
                
                # Device info
                "device": "CPU",
                "noise": "ideal",
                
                # Memory info
                "max_memory_mb": self._mock_backend._max_memory_mb,
                # required_memory_mb: AerSimulator rounds to integer (minimum 1 MB)
                "required_memory_mb": max(1, int(required_mb)),
                
                # Circuit info
                "num_qubits": num_qubits,
                "method": self._mock_backend.options.get("method", "statevector"),
                "active_input_qubits": list(range(num_qubits)),
                # input_qubit_map: AerSimulator uses reversed order [qubit_index, logical_index]
                # Format: [[physical_qubit, logical_qubit], ...] in reverse order
                "input_qubit_map": [[i, i] for i in reversed(range(num_qubits))],
                
                # Optimization info
                "batched_shots_optimization": False,
                "remapped_qubits": False,
                "fusion": {
                    "enabled": True,
                    "threshold": 14,
                    "applied": num_qubits > 14,  # Fusion typically applied for larger circuits
                    "max_fused_qubits": 5  # Constant value used by AerSimulator
                },
                
                # Mock indicator (for debugging)
                "is_mock": True,
                "execution_time_predicted": predicted_time
            }
            
            # Generate all possible Pauli strings for this circuit (for EstimatorV2 compatibility)
            # EstimatorV2 will call result.data(flat_index)[pauli] where pauli is a string like 'ZII...'
            # We'll generate a dict with common Pauli strings and their expectation values
            pauli_expectation_values = self._generate_pauli_expectation_values(circuit, expval)
            
            # Create result data as dict (for Result.from_dict compatibility)
            results.append({
                "success": True,
                "status": "DONE",
                "shots": shots,
                "meas_level": 2,  # Standard measurement level
                "seed_simulator": hash((num_qubits, depth, len(circuit.data))) % (2**32),  # Deterministic seed
                "data": {
                    "counts": self._generate_mock_counts(circuit, expval),
                },
                "header": header,
                "metadata": metadata,
                "time_taken": predicted_time
            })
            
            # Store pauli expectation values for later use (after Result is created)
            if not hasattr(self, '_pauli_evs_store'):
                self._pauli_evs_store = {}
            self._pauli_evs_store[idx] = pauli_expectation_values
        
        # Create Result object compatible with AerSimulator
        result = Result.from_dict({
            "results": results,
            "qobj_id": "mock_qobj",
            "success": True,
            "backend_name": "mock_aer_simulator",
            "backend_version": "0.1.0"
        })
        
        # Make data objects callable for EstimatorV2 compatibility
        # EstimatorV2 calls result.data(flat_index)[pauli], so we need to make data callable
        if hasattr(self, '_pauli_evs_store'):
            for idx, exp_result in enumerate(result.results):
                if idx in self._pauli_evs_store:
                    pauli_evs = self._pauli_evs_store[idx]
                    # Create a callable wrapper around the data object
                    original_data = exp_result.data
                    
                    class CallableDataWrapper:
                        """Wrapper that makes ExperimentResultData callable for EstimatorV2."""
                        def __init__(self, original_data, pauli_evs):
                            self._original = original_data
                            self._pauli_evs = pauli_evs
                            # Copy all attributes from original (especially 'counts')
                            if hasattr(original_data, 'counts'):
                                self.counts = original_data.counts
                        
                        def __call__(self, flat_index):
                            """Called by EstimatorV2 as result.data(flat_index)[pauli]"""
                            # Return a dict that handles missing keys
                            # Use collections.defaultdict-like behavior but with deterministic values
                            from collections import UserDict
                            
                            class PauliDict(UserDict):
                                """Dict that returns a value for missing Pauli strings."""
                                def __init__(self, pauli_evs):
                                    super().__init__(pauli_evs)
                                
                                def __missing__(self, key):
                                    """Called when key is missing - generate deterministic value."""
                                    # If key not found, generate a deterministic value
                                    num_qubits = len(key) if isinstance(key, str) else 0
                                    seed = hash((key, num_qubits)) % 1000000
                                    np.random.seed(seed)
                                    value = np.random.uniform(-1.0, 1.0)
                                    # Cache it
                                    self[key] = value
                                    return value
                            
                            return PauliDict(self._pauli_evs)
                        
                        def __getattr__(self, name):
                            """Delegate to original data object for other attributes"""
                            return getattr(self._original, name)
                        
                        def to_dict(self):
                            """For compatibility"""
                            return self._original.to_dict()
                    
                    # Replace the data object with callable wrapper
                    exp_result.data = CallableDataWrapper(original_data, pauli_evs)
            
            # Clean up
            delattr(self, '_pauli_evs_store')
        
        return result
    
    def _compute_mock_expectation_value(self, circuit: QuantumCircuit) -> float:
        """
        Compute a mock expectation value that's deterministic but circuit-dependent.
        
        This is a simple heuristic - for more realistic mocks, you could:
        - Use a simplified statevector computation
        - Use circuit properties (depth, gates, etc.)
        - Use a learned model
        """
        num_qubits = circuit.num_qubits
        depth = circuit.depth()
        num_gates = len(circuit.data)
        
        # Deterministic but circuit-dependent value
        seed = hash((num_qubits, depth, num_gates)) % 1000000
        np.random.seed(seed)
        expval = np.random.uniform(-1.0, 1.0)
        return float(expval)
    
    def _generate_pauli_expectation_values(self, circuit: QuantumCircuit, base_expval: float) -> dict:
        """
        Generate expectation values for common Pauli strings.
        
        EstimatorV2 calls result.data(flat_index)[pauli] where pauli is a string like 'ZII...'
        This method generates a dict with Pauli strings as keys and expectation values as values.
        
        Args:
            circuit: The quantum circuit
            base_expval: Base expectation value (for Z...I observables)
            
        Returns:
            Dict mapping Pauli strings to expectation values
        """
        num_qubits = circuit.num_qubits
        pauli_evs = {}
        
        # Generate expectation values for common Pauli strings
        # Z...I (measure first qubit) - this is the most common pattern
        z_pauli = 'Z' + 'I' * (num_qubits - 1)
        pauli_evs[z_pauli] = base_expval
        
        # I...I (identity - always 1.0)
        identity_pauli = 'I' * num_qubits
        pauli_evs[identity_pauli] = 1.0
        
        # Generate Z on each qubit individually (covers all single-qubit Z measurements)
        for i in range(num_qubits):
            pauli_str = 'I' * i + 'Z' + 'I' * (num_qubits - i - 1)
            if pauli_str not in pauli_evs:  # Don't overwrite the first one
                # Use a deterministic but varied value
                seed = hash((num_qubits, i, len(circuit.data))) % 1000000
                np.random.seed(seed)
                pauli_evs[pauli_str] = np.random.uniform(-1.0, 1.0)
        
        # X and Y patterns on first few qubits (for more complex observables)
        for i in range(min(num_qubits, 5)):
            # X pattern
            pauli_str = 'I' * i + 'X' + 'I' * (num_qubits - i - 1)
            seed = hash((num_qubits, i, 'X', len(circuit.data))) % 1000000
            np.random.seed(seed)
            pauli_evs[pauli_str] = np.random.uniform(-1.0, 1.0)
            
            # Y pattern
            pauli_str = 'I' * i + 'Y' + 'I' * (num_qubits - i - 1)
            seed = hash((num_qubits, i, 'Y', len(circuit.data))) % 1000000
            np.random.seed(seed)
            pauli_evs[pauli_str] = np.random.uniform(-1.0, 1.0)
        
        return pauli_evs
    
    def _generate_mock_counts(self, circuit: QuantumCircuit, expval: float) -> dict:
        """
        Generate mock measurement counts based on expectation value.
        
        This creates a simple distribution that's consistent with the expectation value.
        """
        num_qubits = circuit.num_qubits
        shots = self._run_options.get("shots", 1024)
        
        # Simple heuristic: create counts that give the expected value
        # For Z measurement on first qubit with expectation value expval:
        # P(0) = (1 + expval) / 2, P(1) = (1 - expval) / 2
        p0 = (1 + expval) / 2
        p0 = max(0.0, min(1.0, p0))  # Clamp to [0, 1]
        
        count_0 = int(shots * p0)
        count_1 = shots - count_0
        
        # Format as Qiskit counts (binary strings)
        counts = {}
        if count_0 > 0:
            counts["0" * num_qubits] = count_0
        if count_1 > 0:
            counts["1" + "0" * (num_qubits - 1)] = count_1
        
        return counts


class MockEstimatorV2:
    """
    Mock EstimatorV2 that uses the execution time model.
    
    This provides the same interface as qiskit_aer.primitives.EstimatorV2
    but uses models instead of actual execution.
    """
    
    def __init__(self, backend: MockAerSimulator):
        self.backend = backend
        
    @classmethod
    def from_backend(cls, backend: MockAerSimulator):
        """Create MockEstimatorV2 from a backend (compatible with EstimatorV2 interface)."""
        return cls(backend)
    
    def run(self, circuits_observables, **kwargs):
        """Run estimation using the model."""
        results = []
        
        for circuit, observable in circuits_observables:
            num_qubits = circuit.num_qubits
            depth = circuit.depth()
            
            # Check memory
            is_feasible, required_mb, max_mb = check_memory_feasibility(
                num_qubits, self.backend.max_memory_mb
            )
            
            if not is_feasible:
                error_msg = (
                    f"ERROR:  [Experiment {len(results)}] Insufficient memory to run circuit "
                    f"using the statevector simulator. "
                    f"Required memory: {required_mb:.0f}M, max memory: {max_mb:.0f}M"
                )
                raise MemoryError(error_msg)
            
            # Predict execution time
            predicted_time = predict_execution_time(
                num_qubits=num_qubits,
                depth=depth,
                reps=2
            )
            
            # Simulate delay
            if self.backend.simulate_delay:
                time.sleep(min(predicted_time, 1.0))
            
            # Mock expectation value
            if self.backend.mock_expectation_value is not None:
                expval = self.backend.mock_expectation_value
            else:
                # Deterministic but circuit-dependent
                seed = hash((num_qubits, depth, len(circuit.data))) % 1000000
                np.random.seed(seed)
                expval = np.random.uniform(-1.0, 1.0)
            
            # Create result object compatible with EstimatorV2
            result_obj = type('Result', (object,), {
                "data": type('Data', (object,), {
                    "evs": np.array([expval])
                })(),
                "metadata": {
                    "execution_time_predicted": predicted_time,
                    "memory_required_mb": required_mb,
                    "memory_feasible": is_feasible
                }
            })()
            results.append(result_obj)
        
        return MockEstimatorResult(results)


class MockEstimatorResult:
    """Mock result object for EstimatorV2."""
    
    def __init__(self, results):
        self._results = results
    
    def result(self):
        """Get results (compatible with EstimatorV2 interface)."""
        return self._results
    
    def __getitem__(self, index):
        """Allow indexing like EstimatorV2 results."""
        return self._results[index]


# Convenience function for Q-Dreamer integration
def create_mock_backend(max_memory_mb: float = 65536.0, simulate_delay: bool = False):
    """
    Create a mock backend for Q-Dreamer optimization.
    
    Args:
        max_memory_mb: Maximum available memory in MB
        simulate_delay: Whether to simulate execution delays
        
    Returns:
        MockAerSimulator instance
    """
    return MockAerSimulator(
        max_memory_mb=max_memory_mb,
        simulate_delay=simulate_delay,
        name="mock_aer_for_qdreamer"
    )

