# src/quex/parser.py
from typing import List, Dict, Any
import openqasm3
from openqasm3 import ast

def parse_qasm_string(qasm_string: str) -> List[Dict[str, Any]]:
    """
    Parses an OpenQASM 3 string into a list of Quex operations.

    Returns a list of dictionaries, where each dict represents a gate
    and the target qubits it acts upon.
    """
    # Parse the string into an Abstract Syntax Tree
    try:
        program = openqasm3.parse(qasm_string)
    except Exception as e:
        raise ValueError(f"Failed to parse OpenQASM: {e}")

    circuit_ops = []

    # Walk the AST to extract gates and their targets
    for statement in program.statements:
        if isinstance(statement, ast.QuantumGate):
            gate_name = statement.name.name

            target_qubits = []
            for arg in statement.arguments:
                if isinstance(arg, ast.IndexedIdentifier):
                    # E.g., q[0] -> name is 'q', index is 0
                    qubit_name = arg.name.name
                    qubit_index = arg.indices[0][0].value
                    target_qubits.append((qubit_name, qubit_index))

            circuit_ops.append({
                "gate": gate_name,
                "targets": target_qubits
            })

    return circuit_ops
