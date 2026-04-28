"""
NumpySimulator
--------------

Module for the numpy simulator backend.
"""

import numpy as np
from quex.gates import STATIC_GATES

class NumpySimulator:
    """
    A high-performance, tensor-network style statevector simulator using NumPy.
    """

    def __init__(self):
        # Look how clean this is now! No more hardcoded dictionaries.
        pass

    def _get_gate_tensor(self, name: str, params: list, num_targets: int) -> np.ndarray:
        """Fetches the matrix from the central registry and reshapes it for tensor contraction."""
        
        # 1. Handle Static Gates (X, H, CX, CZ, etc.)
        if name in STATIC_GATES:
            # STATIC_GATES format: (num_qubits, num_params, numpy_matrix)
            matrix = STATIC_GATES[name][2] 
            return matrix.reshape((2,) * (2 * num_targets))
            
        # 2. Handle Parameterized Gates dynamically
        elif name == "rx":
            theta = params[0]
            matrix = np.array([
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)]
            ], dtype=np.complex128)
            return matrix.reshape((2, 2))
            
        elif name == "ry":
            theta = params[0]
            matrix = np.array([
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)]
            ], dtype=np.complex128)
            return matrix.reshape((2, 2))
            
        elif name == "rz":
            theta = params[0]
            matrix = np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ], dtype=np.complex128)
            return matrix.reshape((2, 2))

        elif name == "p":
            lam = params[0]
            matrix = np.array([
                [1, 0],
                [0, np.exp(1j * lam)]
            ], dtype=np.complex128)
            return matrix.reshape((2, 2))
            
        elif name == "u":
            theta, phi, lam = params
            matrix = np.array([
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
            ], dtype=np.complex128)
            return matrix.reshape((2, 2))
            
        else:
            raise ValueError(f"Gate '{name}' is not supported by NumpySimulator.")

    def run(self, circuit, parameter_binds: dict = None) -> np.ndarray:
        """
        Executes the circuit and returns the final N-dimensional state tensor.
        Accepts an optional dictionary of parameter bindings.
        """
        if parameter_binds is None:
            parameter_binds = {}

        num_qubits = circuit.num_qubits
        if num_qubits == 0:
            return np.array([])

        # 1. Initialize the state to |00...0>
        # A 3-qubit state is shape (2, 2, 2). Only index [0, 0, 0] is 1.0.
        state = np.zeros((2,) * num_qubits, dtype=np.complex128)
        state[(0,) * num_qubits] = 1.0

        # 2. Iterate through the topological operations
        for op in circuit.operations:
            gate_name = op["gate"]

            # --- Parameter Binding part ---
            bound_params = []
            if op["params"]:
                for p in op["params"]:
                    # If the parameter is a string (variable name), look it up in the dict
                    if isinstance(p, str):
                        if p not in parameter_binds:
                            raise ValueError(f"Unbound parameter: '{p}'. Please provide it in parameter_binds.")
                        bound_params.append(float(parameter_binds[p]))
                    else:
                        # It's already a number
                        bound_params.append(p)
            # -------------------------------

            # Extract clean integer targets (e.g., [0, 1])
            targets = [t[1] for t in op["targets"] if t[1] is not None]
            k = len(targets)  # How many qubits this gate touches

            gate_tensor = self._get_gate_tensor(gate_name, bound_params, k)

            # --- THE TENSOR CONTRACTION MAGIC ---

            # Step A: gate_tensor has 2*k axes. The last 'k' axes are the inputs.
            gate_input_axes = list(range(k, 2 * k))

            # Step B: Multiply the gate's input axes against the state's target axes
            state = np.tensordot(gate_tensor, state, axes=(gate_input_axes, targets))

            # Step C: tensordot dumps the new output axes at the very front of the array (indices 0 to k-1).
            # We must move them back to their proper physical qubit slots.
            state = np.moveaxis(state, source=list(range(k)), destination=targets)

        return state
