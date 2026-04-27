# src/quex/gates.py
"""
Gates
-----

This module has gate definitions used
throughout the quex package.

"""

import numpy as np

# A centralized dictionary of all supported static gates.
# Format: { "gate_name": (num_qubits, num_params, numpy_matrix) }
STATIC_GATES = {
    "x": (1, 0, np.array([[0, 1], [1, 0]], dtype=np.complex128)),
    "y": (1, 0, np.array([[0, -1j], [1j, 0]], dtype=np.complex128)),
    "z": (1, 0, np.array([[1, 0], [0, -1]], dtype=np.complex128)),
    "h": (1, 0, np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)),
    "s": (1, 0, np.array([[1, 0], [0, 1j]], dtype=np.complex128)),
    
    # 2-Qubit gates are stored in their native 4x4 matrix form here.
    "cx": (2, 0, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)),
    "cz": (2, 0, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=np.complex128)),
    "ccx": (3, 0, np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ], dtype=np.complex128))
}

# Define the signatures for parameterized gates
PARAMETRIZED_GATES = {
    "rx": (1, 1),  # (num_qubits, num_params)
    "ry": (1, 1),
    "rz": (1, 1),
}

def get_supported_gates():
    """Returns a list of all gate names supported by Quex."""
    return list(STATIC_GATES.keys()) + list(PARAMETRIZED_GATES.keys())