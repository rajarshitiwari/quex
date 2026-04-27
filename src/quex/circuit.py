# src/quex/core/circuit.py
from typing import Any, Dict, List, Optional

import networkx as nx

from quex.parser import parse_qasm_string


class Circuit:
    """
    The internal, backend-agnostic representation of a quantum circuit in Quex.
    """

    def __init__(self, num_qubits: int):
        """
        The simplest state: Initialize a blank circuit with N qubits.
        """
        self.num_qubits = num_qubits

        # The flat list of operations (Gate Name, Params, Targets)
        self.operations: List[Dict[str, Any]] = []

        # The dependency graph (DAG) for execution routing
        self.dag = nx.DiGraph()

    def add_operation(self, gate: str, targets: List[tuple], params: Optional[List[float]] = None):
        """
        Programmatically add a gate to the circuit.
        """
        if params is None:
            params = []

        op = {
            "gate": gate.lower(),
            "params": params,
            "targets": targets
        }
        self.operations.append(op)
        self._update_dag(op)

    def _update_dag(self, operation: Dict[str, Any]):
        """
        Updates the NetworkX DAG dynamically as operations are added.
        (You can migrate the logic we wrote in the Jupyter notebook here later).
        """
        pass

    @classmethod
    def from_qasm(cls, qasm_string: str) -> "Circuit":
        """
        Factory method: Builds a Quex Circuit directly from an OpenQASM 3 string.
        """
        # 1. Use the parser to get the intermediate representation
        parsed_ops = parse_qasm_string(qasm_string)

        # 2. Figure out the total number of qubits needed
        # (This is a simplified way to find max qubit index + 1)
        max_qubit_index = -1
        for op in parsed_ops:
            for target in op["targets"]:
                if target[1] is not None and target[1] > max_qubit_index:
                    max_qubit_index = target[1]

        num_qubits = max_qubit_index + 1 if max_qubit_index >= 0 else 0

        # 3. Create the simple, empty circuit
        circuit = cls(num_qubits=num_qubits)

        # 4. Populate it with the parsed operations
        for op in parsed_ops:
            circuit.add_operation(
                gate=op["gate"],
                targets=op["targets"],
                params=op["params"]
            )

        return circuit
