# src/quex/parser.py
from typing import Any, Dict

import openqasm3
from openqasm3 import ast


def parse_qasm_string(qasm_string: str) -> Dict[str, Any]:
    """
    Parses an OpenQASM 3 string into a list of Quex operations.

    Returns a list of dictionaries, where each dict represents a gate
    and the target qubits it acts upon.
    """
    try:
        program = openqasm3.parse(qasm_string)
    except Exception as e:
        raise ValueError(f"Failed to parse OpenQASM: {e}")

    registers = {}
    circuit_ops = []

    for statement in program.statements:
        # NEW: Catch the register declarations!
        if isinstance(statement, ast.QubitDeclaration):
            reg_name = statement.qubit.name
            reg_size = statement.size.value
            registers[reg_name] = reg_size

        elif isinstance(statement, ast.QuantumGate):
            gate_name = statement.name.name

            # 1. Extract Target Qubits (from statement.qubits)
            target_qubits = []
            for q in statement.qubits:
                if isinstance(q, ast.IndexedIdentifier):
                    qubit_name = q.name.name
                    qubit_index = q.indices[0][0].value
                    target_qubits.append((qubit_name, qubit_index))

            # 2. Extract Parameters/Angles (from statement.arguments)
            params = []
            if statement.arguments:
                for arg in statement.arguments:
                    if hasattr(arg, "value"):
                        # Catches both ast.RealLiteral and ast.IntegerLiteral
                        params.append(arg.value)
                    elif hasattr(arg, "name"):  # Catch string variables (Identifiers)
                        params.append(arg.name)

            circuit_ops.append({"gate": gate_name, "params": params, "targets": target_qubits})

    # Return both!
    return {"registers": registers, "operations": circuit_ops}
