"""
NumpySimulator
--------------

Module for the numpy simulator backend.
"""

import numpy as np


class NumpySimulator:
    """
    A high-performance, tensor-network style statevector simulator using NumPy.
    """

    def __init__(self):
        # Build the 8x8 Toffoli matrix and reshape it to (2,2,2,2,2,2)
        ccx_mat = np.eye(8, dtype=np.complex128)
        # In a Toffoli, if controls are |11> (binary 110=6, 111=7), apply X to the target
        ccx_mat[6, 6] = 0
        ccx_mat[7, 7] = 0
        ccx_mat[6, 7] = 1
        ccx_mat[7, 6] = 1

        self._static_gates = {
            # 1-Qubit Gates
            "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
            "h": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
            # 2-Qubit gates must be reshaped into (2, 2, 2, 2) tensors
            # [output_q0, output_q1, input_q0, input_q1]
            "cx": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128).reshape(2, 2, 2, 2),
            # 3-Qubit Gates (reshaped to 2, 2, 2, 2, 2, 2)
            "ccx": ccx_mat.reshape(2, 2, 2, 2, 2, 2),
            "toffoli": ccx_mat.reshape(2, 2, 2, 2, 2, 2),  # Alias for ccx
        }

    def _get_gate_tensor(self, name: str, params: list, num_targets: int) -> np.ndarray:
        """Retrieves or calculates the gate matrix and reshapes it for tensor math."""
        if name in self._static_gates:
            return self._static_gates[name]

        # Dynamically build parameterized rotation gates
        if name == "rx":
            theta = params[0]
            return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex128)

        elif name == "ry":
            theta = params[0]
            return np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex128)

        elif name == "rz":
            theta = params[0]
            return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128)

        raise NotImplementedError(f"Gate '{name}' is not yet supported by NumpySimulator.")

    def run(self, circuit) -> np.ndarray:
        """Executes the circuit and returns the final N-dimensional state tensor."""
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
            params = op["params"]

            # Extract clean integer targets (e.g., [0, 1])
            targets = [t[1] for t in op["targets"] if t[1] is not None]
            k = len(targets)  # How many qubits this gate touches

            gate_tensor = self._get_gate_tensor(gate_name, params, k)

            # --- THE TENSOR CONTRACTION MAGIC ---

            # Step A: gate_tensor has 2*k axes. The last 'k' axes are the inputs.
            gate_input_axes = list(range(k, 2 * k))

            # Step B: Multiply the gate's input axes against the state's target axes
            state = np.tensordot(gate_tensor, state, axes=(gate_input_axes, targets))

            # Step C: tensordot dumps the new output axes at the very front of the array (indices 0 to k-1).
            # We must move them back to their proper physical qubit slots.
            state = np.moveaxis(state, source=list(range(k)), destination=targets)

        return state
