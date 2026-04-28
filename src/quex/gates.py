# src/quex/gates.py
"""
Gates
-----

This module has gate definitions used
throughout the quex package.

It contains a centralized dictionary of all supported static gates,
and one for parameterised gates.
"""

import numpy as np

# Commonly used constants
PI = np.pi
SQRT2 = np.sqrt(2)

# Format: { "gate_name": (num_qubits, num_params, numpy_matrix) }
STATIC_GATES = {
    # --- 1-Qubit Pauli Gates ---
    "id": (1, 0, np.array([[1, 0], [0, 1]], dtype=np.complex128)),
    "x":  (1, 0, np.array([[0, 1], [1, 0]], dtype=np.complex128)),
    "y":  (1, 0, np.array([[0, -1j], [1j, 0]], dtype=np.complex128)),
    "z":  (1, 0, np.array([[1, 0], [0, -1]], dtype=np.complex128)),
    
    # --- 1-Qubit Clifford & Phase Gates ---
    "h":  (1, 0, np.array([[1, 1], [1, -1]], dtype=np.complex128) / SQRT2),
    "s":  (1, 0, np.array([[1, 0], [0, 1j]], dtype=np.complex128)),
    "sdg":(1, 0, np.array([[1, 0], [0, -1j]], dtype=np.complex128)),
    "t":  (1, 0, np.array([[1, 0], [0, np.exp(1j * PI / 4)]], dtype=np.complex128)),
    "tdg":(1, 0, np.array([[1, 0], [0, np.exp(-1j * PI / 4)]], dtype=np.complex128)),
    
    # Square root of X (used heavily by IBM hardware)
    "sx": (1, 0, np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=np.complex128) / 2),
    "sxdg":(1, 0, np.array([[1-1j, 1+1j], [1+1j, 1-1j]], dtype=np.complex128) / 2),

    # --- 2-Qubit Entangling Gates ---
    "cx": (2, 0, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)),
    
    "cy": (2, 0, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0]
    ], dtype=np.complex128)),

    "cz": (2, 0, np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=np.complex128)),
    
    "swap": (2, 0, np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)),

    # --- 3-Qubit Gates ---
    # CCX / Toffoli Gate (Only flips the last qubit if first two are 1)
    "ccx": (3, 0, np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=np.complex128)),
}

# Define the signatures for parameterized gates. 
# The actual math for these remains in the NumpySimulator backend, 
# since they have to be evaluated dynamically at runtime!
PARAMETRIZED_GATES = {
    # Gate: (num_qubits, num_params)
    "rx": (1, 1), 
    "ry": (1, 1),
    "rz": (1, 1),
    "p":  (1, 1),  # Phase gate (similar to RZ but adds global phase)
    "u":  (1, 3),  # Universal gate (theta, phi, lambda)
}

def get_supported_gates():
    """Returns a list of all gate names supported by Quex."""
    return list(STATIC_GATES.keys()) + list(PARAMETRIZED_GATES.keys())
