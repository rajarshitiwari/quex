"""
Random Circuits
---------------


Example
-------
.. jupyter-execute::

    if __name__ == "__main__":
        print(random_qasm(num_qubits=3, depth=5))
        print(random_qiskit(num_qubits=3, depth=5))

"""

import random


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

    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{num_qubits}] q;"
    ]

    single_qubit_gates = ['h', 'x', 'rx', 'rz']

    for _ in range(depth):
        gate = random.choice(single_qubit_gates)
        target = random.randint(0, num_qubits - 1)

        # Handle parameterized gates
        if gate in ['rx', 'rz']:
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
        from qiskit.circuit.random import random_circuit
        from qiskit import qasm3
    except ImportError:
        raise ImportError(
            "Qiskit is not installed. To use this function, "
            "install dev dependencies: `uv add --dev qiskit`"
        )

    qc = random_circuit(num_qubits=4, depth=5, measure=False)
    # Export directly to an OpenQASM 3 string
    qasm_string = qasm3.dumps(qc)
    return qasm_string

# Example usage:
if __name__ == "__main__":
    print(random_qasm(num_qubits=3, depth=5))
    print(random_qiskit(num_qubits=3, depth=5))
