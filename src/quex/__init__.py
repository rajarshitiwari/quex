"""
Quex
----

Quantum circuit execution

"""

import importlib.metadata

from quex.backends.numpy_sim import NumpySimulator
from quex.circuit import Circuit
from quex.parser import parse_qasm_string
from quex.qrandom import random_ansatz, random_ansatz_P, random_ansatz_U, random_qasm, random_qiskit
from quex.vis import draw_dag, draw_qiskit, draw_structured_dag

__version__ = importlib.metadata.version("quex")
version = __version__

__all__ = [
    "Circuit",
    "draw_dag",
    "draw_qiskit",
    "draw_structured_dag",
    "NumpySimulator",
    "parse_qasm_string",
    "random_ansatz",
    "random_ansatz_U",
    "random_ansatz_P",
    "random_qasm",
    "random_qiskit",
    "version",
]
