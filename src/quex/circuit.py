# src/quex/circuit.py
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
        # Internal tracker to know where to draw edges
        self._last_seen = {}

    def add_operation(self, gate: str, targets: List[tuple], params: Optional[List[float]] = None):
        """
        Programmatically add an operation to the circuit.
        """
        if params is None:
            params = []

        op = {"gate": gate.lower(), "params": params, "targets": targets}
        self.operations.append(op)

        # Update the DAG
        op_index = len(self.operations) - 1
        self._update_dag(op_index, op)

    def _update_dag(self, index: int, operation: Dict[str, Any]):
        """Dynamically builds the execution dependency graph."""
        gate_name = operation["gate"].upper()
        targets = operation["targets"]

        node_id = f"{index}: {gate_name}"
        self.dag.add_node(node_id, gate=gate_name, targets=targets)

        # Draw edges from the previous gate on these qubits to this new gate
        for qubit in targets:
            if qubit in self._last_seen:
                self.dag.add_edge(self._last_seen[qubit], node_id, label=f"q[{qubit[1]}]")
            self._last_seen[qubit] = node_id

    @classmethod
    def from_qasm(cls, qasm_string: str) -> "Circuit":
        """
        Builds a Quex Circuit directly from an OpenQASM 3 string.
        """
        # 1. Use the parser to get the intermediate representation
        parsed_ops = parse_qasm_string(qasm_string)

        # 2. Determine total qubits dynamically
        max_qubit_index = -1
        for op in parsed_ops:
            for target in op["targets"]:
                if target[1] is not None and target[1] > max_qubit_index:
                    max_qubit_index = target[1]

        num_qubits = max_qubit_index + 1 if max_qubit_index >= 0 else 0

        # 3. Construct and populate
        circuit = cls(num_qubits=num_qubits)
        for op in parsed_ops:
            circuit.add_operation(gate=op["gate"], targets=op["targets"], params=op["params"])

        return circuit

    def to_text_diagram(self) -> str:
        """
        Generates a native, zero-dependency ASCII/Unicode representation
        of the quantum circuit.
        """
        if self.num_qubits == 0:
            return "Empty Circuit"

        wires = [f"q[{i}]: ──" for i in range(self.num_qubits)]

        for op in self.operations:
            gate_name = op["gate"].upper()

            # IMPROVEMENT 1: Format parameterized gates to 2 decimal places
            if op["params"]:
                params_str = ",".join(str(round(p, 2)) for p in op["params"])
                gate_str = f"{gate_name}({params_str})"
            else:
                gate_str = gate_name

            targets = [t[1] for t in op["targets"] if t[1] is not None]
            if not targets:
                continue

            col_width = len(gate_str) + 4

            if len(targets) == 1:
                t = targets[0]
                for i in range(self.num_qubits):
                    if i == t:
                        wires[i] += f"[{gate_str}]".center(col_width, "─")
                    else:
                        wires[i] += "─" * col_width

            elif len(targets) == 2:
                ctrl, tgt = targets[0], targets[1]
                top, bottom = min(ctrl, tgt), max(ctrl, tgt)

                for i in range(self.num_qubits):
                    if i == ctrl:
                        wires[i] += "■".center(col_width, "─")
                    elif i == tgt:
                        symbol = "X" if gate_name == "CX" else f"[{gate_str}]"
                        wires[i] += symbol.center(col_width, "─")
                    elif top < i < bottom:
                        wires[i] += "│".center(col_width, "─")
                    else:
                        wires[i] += "─" * col_width

            # IMPROVEMENT 2: Handle 3-qubit gates (like CCX / Toffoli)
            elif len(targets) == 3:
                ctrl1, ctrl2, tgt = targets[0], targets[1], targets[2]
                top, bottom = min(targets), max(targets)

                for i in range(self.num_qubits):
                    if i in (ctrl1, ctrl2):
                        wires[i] += "■".center(col_width, "─")
                    elif i == tgt:
                        symbol = "X" if gate_name in ["CCX", "TOFFOLI"] else f"[{gate_str}]"
                        wires[i] += symbol.center(col_width, "─")
                    elif top < i < bottom:
                        wires[i] += "│".center(col_width, "─")
                    else:
                        wires[i] += "─" * col_width

        return "\n".join(wires)

    def __repr__(self) -> str:
        """Allows the circuit to draw itself automatically in standard REPLs."""
        return self.to_text_diagram()

    def __str__(self) -> str:
        """Allows `print(circuit)` to output the text diagram."""
        return self.to_text_diagram()
