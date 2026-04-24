"""
Random Circuits
---------------
"""

import random


def generate_random_qasm(num_qubits: int, depth: int) -> str:
    """Generates a random OpenQASM 3 string with standard gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    depth : int
        Circuit depth

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


# Example usage:
if __name__ == "__main__":
    print(generate_random_qasm(num_qubits=3, depth=5))
