"""
Random Circuits
---------------


Example
-------

.. jupyter-execute::

    import quex as qx
    qc_static = qx.random_ansatz(num_qubits=6, depth=6)
    print("Static Random Circuit:")
    print(qc_static)
    print(f"Compiler Operations Depth: {qc_static.depth}")

"""

import math
import random

from quex.circuit import Circuit


def random_ansatz_U(num_qubits: int, depth: int) -> Circuit:
    """
    Generates a Hardware-Efficient Ansatz using Haar-uniform random rotations.
    """
    qc = Circuit(num_qubits)

    for d in range(depth):
        # 1. Layer of Uniformly Random Bloch Sphere Rotations
        for i in range(num_qubits):
            # Uniformly distributed phi and lambda
            phi = random.uniform(0, 2 * math.pi)
            lam = random.uniform(0, 2 * math.pi)

            # The arccos trick to prevent pole-bunching on the sphere
            u = random.uniform(0, 1)
            theta = math.acos(1 - 2 * u)

            qc.add_operation("u", i, params=[theta, phi, lam])

        # 2. Layer of Linear Entanglement
        offset = d % 2
        for i in range(offset, num_qubits - 1, 2):
            qc.add_operation("cx", [i, i + 1])

    return qc


def random_ansatz(num_qubits: int, depth: int) -> Circuit:
    """
    Generates a 'Hardware-Efficient Ansatz' circuit.
    This consists of alternating layers of random rotations and linear entanglement.
    It is the standard template for Quantum Machine Learning and pseudo-random states.
    """
    qc = Circuit(num_qubits)

    for d in range(depth):
        # 1. Layer of Random Rotations (Rx, Ry, Rz)
        for i in range(num_qubits):
            gate = random.choice(["rx", "ry", "rz"])
            angle = random.uniform(0, 3.14159 * 2)  # Random angle between 0 and 2pi
            qc.add_operation(gate, i, params=[angle])

        # 2. Layer of Linear Entanglement (CX gates connecting neighbors)
        # We alternate even/odd pairs to create a "brickwork" pattern
        offset = d % 2
        for i in range(offset, num_qubits - 1, 2):
            qc.add_operation("cx", [i, i + 1])

    return qc


def random_ansatz_P(num_qubits: int, depth: int, parameterized: bool = True) -> Circuit:
    """
    Generates a Hardware-Efficient Ansatz.
    If parameterized=True, it builds a template with string variables and
    saves the initial random angles into qc.parameters (Perfect for ML!).
    """
    qc = Circuit(num_qubits)
    param_idx = 0

    for d in range(depth):
        # 1. Layer of Uniformly Random Bloch Sphere Rotations
        for i in range(num_qubits):
            # Calculate the mathematically uniform random angles
            phi_val = random.uniform(0, 2 * math.pi)
            lam_val = random.uniform(0, 2 * math.pi)
            u = random.uniform(0, 1)
            theta_val = math.acos(1 - 2 * u)

            if parameterized:
                # Create variable names: t_0, p_0, l_0
                t_str, p_str, l_str = f"t_{param_idx}", f"p_{param_idx}", f"l_{param_idx}"
                param_idx += 1

                # Add the operations as Strings (Late Binding)
                qc.add_operation("u", i, params=[t_str, p_str, l_str])

                # Store the actual numbers in the circuit's state!
                qc.parameters[t_str] = theta_val
                qc.parameters[p_str] = phi_val
                qc.parameters[l_str] = lam_val
            else:
                # Add the operations strictly as Floats (Static binding)
                qc.add_operation("u", i, params=[theta_val, phi_val, lam_val])

        # 2. Layer of Linear Entanglement
        offset = d % 2
        for i in range(offset, num_qubits - 1, 2):
            qc.add_operation("cx", [i, i + 1])

    return qc


def random_qasm(num_qubits: int = 3, depth: int = 3) -> str:
    """Generates a random OpenQASM 3 string with standard gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, by default 3
    depth : int
        Circuit depth, by default 3

    Returns
    -------
    str
        A random quantum circuit in OpenQASM 3 format.
    """

    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{num_qubits}] q;"]

    single_qubit_gates = ["h", "x", "rx", "rz"]

    for _ in range(depth):
        gate = random.choice(single_qubit_gates)
        target = random.randint(0, num_qubits - 1)

        # Handle parameterized gates
        if gate in ["rx", "rz"]:
            angle = round(random.uniform(0.0, 3.14159), 4)
            lines.append(f"{gate}({angle}) q[{target}];")
        else:
            lines.append(f"{gate} q[{target}];")

        # Toss in a 2-qubit CNOT gate randomly (if we have >1 qubit)
        if num_qubits > 1 and random.random() > 0.5:
            ctrl = random.randint(0, num_qubits - 1)
            tgt = random.choice([i for i in range(num_qubits) if i != ctrl])
            lines.append(f"cx q[{ctrl}], q[{tgt}];")

    return "\n".join(lines)


def random_qiskit(num_qubits: int = 3, depth: int = 3) -> str:
    """
    Generate a random circuit with num_qubits and a depth, measure=False
    prevents it from adding classical registers just yet.

    Parameters
    ----------
    num_qubits : int, optional
        Number of qubits, by default 3
    depth : int, optional
        Circuit depth, by default 3

    Returns
    -------
    str
        A random quantum circuit in OpenQASM 3 format.
    """
    try:
        from qiskit import qasm3
        from qiskit.circuit.random import random_circuit
    except ImportError:
        raise ImportError("Qiskit is not installed. To use this function, install dev dependencies: `uv add --dev qiskit`")

    qc = random_circuit(num_qubits=4, depth=5, measure=False)
    # Export directly to an OpenQASM 3 string
    qasm_string = qasm3.dumps(qc)
    return qasm_string
