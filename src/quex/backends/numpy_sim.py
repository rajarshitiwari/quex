"""
NumpySimulator
--------------

Module for the numpy simulator backend.
"""

import numpy as np
from numpy.typing import ArrayLike

from quex.backends.base import Simulator
from quex.gates import STATIC_GATES


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

    def __init__(self):
        super().__init__()
        self.xp = np

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

        # 1. Initialize the state to |00...0>, or use existing
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

        # 2. Iterate through the topological operations
        for op in target_circuit.operations:
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
            gate_tensor = get_gate_tensor(gate_name, bound_params, k)

            # --- THE TENSOR CONTRACTION MAGIC ---

            # Step A: gate_tensor has 2*k axes. The last 'k' axes are the inputs.
            gate_input_axes = list(range(k, 2 * k))

            # Step B: Multiply the gate's input axes against the state's target axes
            state = self.xp.tensordot(gate_tensor, state, axes=(gate_input_axes, targets))

            # Step C: tensordot dumps the new output axes at the very front of the array (indices 0 to k-1).
            # We must move them back to their proper physical qubit slots.
            state = self.xp.moveaxis(state, source=list(range(k)), destination=targets)
            # --- NEW: Save the result to the Circuit object ---

        target_circuit.state = state

        return target_circuit.statevector
