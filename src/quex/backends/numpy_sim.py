"""
NumpySimulator
--------------

Module for the numpy simulator backend.
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Literal

from quex.backends.base import Simulator
from quex.gates import STATIC_GATES
from threadpoolctl import threadpool_limits

# --- 1. Independent Matrix Generators ---
def _gen_rx(params: list) -> np.ndarray:
    theta = params[0]
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex128)


def _gen_ry(params: list) -> np.ndarray:
    theta = params[0]
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex128)


def _gen_rz(params: list) -> np.ndarray:
    theta = params[0]
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128)


def _gen_p(params: list) -> np.ndarray:
    lam = params[0]
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128)


def _gen_u(params: list) -> np.ndarray:
    theta, phi, lam = params
    return np.array(
        [[np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)], [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]], dtype=np.complex128
    )


# --- 2. The O(1) Dispatch Table ---
PARAM_GENERATORS = {
    "rx": _gen_rx,
    "ry": _gen_ry,
    "rz": _gen_rz,
    "p": _gen_p,
    "u": _gen_u,
}

# --- NEW: The Global Tensor Cache ---
# Pre-allocate and pre-reshape all static gates ONCE on module load.
# This prevents calling .reshape() millions of times in the simulation loop.
_TENSOR_CACHE = {}
for gate_name, (num_qubits, num_params, matrix) in STATIC_GATES.items():
    _TENSOR_CACHE[gate_name] = matrix.reshape((2,) * (2 * num_qubits))


# --- 3. The Pure Module-Level Function (Updated) ---
def get_gate_tensor(name: str, params: list, num_targets: int) -> np.ndarray:
    """Fetches or dynamically generates the requested gate matrix."""

    # Fast-path: O(1) lookup returns a pure memory reference! No reshaping needed.
    if name in _TENSOR_CACHE:
        return _TENSOR_CACHE[name]

    # Dynamic generation for parameterized gates
    if name in PARAM_GENERATORS:
        matrix = PARAM_GENERATORS[name](params)
        return matrix.reshape((2, 2))
    raise ValueError(f"Gate '{name}' is not supported yet by NumpySimulator.")


# --- 4. The Refined Class ---
class NumpySimulator(Simulator):
    """
    A high-performance, tensor-network style statevector simulator using NumPy.
    """

    def __init__(self,
                 num_threads: int = None,
                 exec_mode: Literal["sequential", "fused"] = "sequential",
                 max_fused_qubits: int = 4
                 ):
        super().__init__()
        self.xp = np
        self.num_threads = num_threads
        self.exec_mode = exec_mode
        self._run_ops = {
            "sequential": self._run_sequential,
            "fused": self._run_fused
        }
        self.max_fused_qubits = max_fused_qubits

    def _allocate_initial_state(self, num_qubits: int) -> ArrayLike:
        """Allocates the default |00...0> state (Overridable by JAX)."""
        state = self.xp.zeros((2,) * num_qubits, dtype=self.xp.complex128)
        state[(0,) * num_qubits] = 1.0
        return state

    def _get_backend_tensor(self, name: str, params: list, num_targets: int):
        """
        Bridge method: Fetches the CPU tensor from the module-level function.
        Subclasses (like CuPy) will override this to intercept the data.
        """
        # Just call your top-level function directly
        return get_gate_tensor(name, params, num_targets)

    def _bind_params(self, params: list, final_binds: dict) -> list:
        """Helper method to resolve symbolic parameters into floats."""
        if not params:
            return []
            
        bound_params = []
        for p in params:
            # If the parameter is a string (variable name), look it up in the dict
            if isinstance(p, str):
                if p not in final_binds:
                    raise ValueError(f"Unbound parameter: '{p}'. Set it via qc.parameters or pass it to run().")
                bound_params.append(float(final_binds[p]))
            else:
                # It's already a number
                bound_params.append(p)
                
        return bound_params

    def _run_sequential(self, circuit, state, final_binds):
        """The original gate-by-gate loop."""
        for op in circuit.operations:
            gate_tensor = self._get_backend_tensor(op["gate"], op["params"], len(op["targets"]))
            # ... tensordot and moveaxis ...
                # 2. Iterate through the topological operations
        for op in circuit.operations:
            gate_name = op["gate"]

            # --- Parameter Binding part ---
            bound_params = []
            if op["params"]:
                for p in op["params"]:
                    # If the parameter is a string (variable name), look it up in the dict
                    if isinstance(p, str):
                        if p not in final_binds:
                            raise ValueError(f"Unbound parameter: '{p}'. Set it via qc.parameters or pass it to run().")
                        bound_params.append(float(final_binds[p]))
                    else:
                        # It's already a number
                        bound_params.append(p)
            # -------------------------------

            # Extract clean integer targets (e.g., [0, 1])
            targets = [t[1] for t in op["targets"] if t[1] is not None]
            k = len(targets)  # How many qubits this gate touches

            # Call the module-level function
            gate_tensor = self._get_backend_tensor(gate_name, bound_params, k)

            # --- THE TENSOR CONTRACTION MAGIC ---

            # Step A: gate_tensor has 2*k axes. The last 'k' axes are the inputs.
            gate_input_axes = list(range(k, 2 * k))

            # Step B: Multiply the gate's input axes against the state's target axes
            state = self.xp.tensordot(gate_tensor, state, axes=(gate_input_axes, targets))

            # Step C: tensordot dumps the new output axes at the very front of the array (indices 0 to k-1).
            # We must move them back to their proper physical qubit slots.
            state = self.xp.moveaxis(state, source=list(range(k)), destination=targets)
            # --- NEW: Save the result to the Circuit object ---

        return state

    def _run_fused(self, target_circuit, state, final_binds):
        """High-performance layer-by-layer execution using Blocked Gate Fusion."""
        
        # The ultimate sweet spot for CPU/GPU L1 Cache is usually 3 to 5 qubits.
        # We can expose this to the user later, but 4 is a very safe default.
        max_fused_qubits = self.max_fused_qubits

        for layer in target_circuit.layers:
            if not layer:
                continue

            chunk_matrix = None
            chunk_targets = []
            chunk_k = 0

            for op in layer:
                # ... resolve parameters ...
                bound_params = []
                if op["params"]:
                    for p in op["params"]:
                        if isinstance(p, str):
                            if p not in final_binds:
                                raise ValueError(f"Unbound parameter: '{p}'")
                            bound_params.append(float(final_binds[p]))
                        else:
                            bound_params.append(p)
                
                targets = [t[1] for t in op["targets"] if t[1] is not None]
                k = len(targets)

                # Fetch and flatten the single gate tensor
                gate_tensor = self._get_backend_tensor(op["gate"], bound_params, k)
                gate_matrix = gate_tensor.reshape((2**k, 2**k))

                # --- THE CHUNKING LOGIC ---
                # If adding this gate exceeds our budget, execute the CURRENT chunk first!
                if chunk_k + k > max_fused_qubits and chunk_matrix is not None:
                    
                    # 1. Reshape and fire the tensordot for the accumulated chunk
                    chunk_tensor = chunk_matrix.reshape((2,) * (2 * chunk_k))
                    gate_input_axes = list(range(chunk_k, 2 * chunk_k))
                    
                    state = self.xp.tensordot(chunk_tensor, state, axes=(gate_input_axes, chunk_targets))
                    state = self.xp.moveaxis(state, source=list(range(chunk_k)), destination=chunk_targets)
                    
                    # 2. Reset the chunk to empty
                    chunk_matrix = None
                    chunk_targets = []
                    chunk_k = 0

                # Accumulate the current gate into the chunk
                if chunk_matrix is None:
                    chunk_matrix = gate_matrix
                else:
                    chunk_matrix = self.xp.kron(chunk_matrix, gate_matrix)
                
                chunk_targets.extend(targets)
                chunk_k += k

            # --- FLUSH THE REMAINDER ---
            # After the loop, if there is a partially filled chunk left over, execute it.
            if chunk_matrix is not None:
                chunk_tensor = chunk_matrix.reshape((2,) * (2 * chunk_k))
                gate_input_axes = list(range(chunk_k, 2 * chunk_k))
                
                state = self.xp.tensordot(chunk_tensor, state, axes=(gate_input_axes, chunk_targets))
                state = self.xp.moveaxis(state, source=list(range(chunk_k)), destination=chunk_targets)
                
        return state

    def run(self, circuit=None, parameter_binds: dict = None, initial_state: ArrayLike = None) -> ArrayLike:
        """
        Executes the circuit and returns the final N-dimensional state tensor.
        Accepts an optional dictionary of parameter bindings.
        Allows passing a circuit directly, OR fallback to the attached circuit.
        """
        target_circuit = circuit or self.circuit

        if target_circuit is None:
            raise ValueError("No circuit provided to run, and no circuit attached to Simulator.")

        # --- NEW: Merge Circuit parameters with temporary binds ---
        # 1. Grab the circuit's inherent parameters (acting as the baseline)
        final_binds = getattr(target_circuit, "parameters", {}).copy()

        # 2. If the user passed explicit binds to this run(), they overwrite the baseline
        if parameter_binds:
            final_binds.update(parameter_binds)

        num_qubits = target_circuit.num_qubits
        if num_qubits == 0:
            return self.xp.array([])
        
        if num_qubits == 0:
            return self.xp.array([])

        # 3. Initialize the state to |00...0>, or use existing
        # --- NEW: State Injection Logic ---
        # Below logic needs to be revisited.
        if initial_state is not None:
            # We MUST copy it so we don't accidentally mutate the previous circuit's memory!
            state = initial_state.copy()
            # Ensure it's reshaped to the tensor format our engine expects
            state = state.reshape((2,) * num_qubits)
        else:
            # Default to all-zeros |00...0>
            # Example: A 3-qubit state is shape (2, 2, 2). Only index [0, 0, 0] is 1.0.
            state = self._allocate_initial_state(num_qubits)

        state = self._run_ops[self.exec_mode](target_circuit, state, final_binds)
        if self.exec_mode not in self._run_ops:
            raise ValueError(f"Unknown exec_mode: '{self.exec_mode}'. Use 'sequential' or 'fused'.")

        target_circuit.state = state

        return target_circuit.statevector
