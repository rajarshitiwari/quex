# src/quex/weave.py

import quex as qx


def _find_spanning_gates(qc, boundary_qubit: int) -> list:
    """
    Identifies multi-qubit gates that cross the horizontal boundary
    between top and bottom sub-circuits.

    The boundary divides the circuit into:
    - Top circuit: qubits 0 to boundary_qubit - 1
    - Bottom circuit: qubits boundary_qubit to N - 1

    Returns:
        A list of tuples containing the original index and the operation dictionary:
        [(operation_index, operation_dict), ...]
    """
    spanning_gates = []

    for idx, op in enumerate(qc.operations):
        targets = op["targets"]

        # 1. Performance Optimization:
        # Single-qubit gates cannot span a boundary. Skip them instantly.
        if isinstance(targets, list) and len(targets) > 1:
            # 2. Robust Extraction:
            # Safely extract integer indices, handling both [0, 1] and [('q', 0), ('q', 1)]
            indices = [t if isinstance(t, int) else t[1] for t in targets]

            # 3. Boundary Check:
            # Does this gate touch qubits on both sides of the cut?
            has_top = any(i < boundary_qubit for i in indices)
            has_bottom = any(i >= boundary_qubit for i in indices)

            if has_top and has_bottom:
                spanning_gates.append((idx, op))

    return spanning_gates


class HybridCircuit:
    """
    Orchestrates the fragmentation, simulation, and recombination
    of large quantum circuits.
    """

    def __init__(self, master_circuit: qx.Circuit):
        self.master_circuit = master_circuit
        self.sub_circuits = {}
        self.bridges = []

    def cleave(self, boundary_qubit: int):
        """
        MOVED FROM CIRCUIT.PY:
        Splits the master circuit into top and bottom chunks at the boundary.
        Cleaves the circuit horizontally into two independent sub-circuits.
        Top circuit: qubits 0 to boundary_qubit - 1
        Bottom circuit: qubits boundary_qubit to N - 1

        Returns:
            qc_top (Circuit): The upper sub-circuit.
            qc_bottom (Circuit): The lower sub-circuit.
            spanning_gates (list): The metadata of the gates that were cut.
        """
        # FIXED: Use master_circuit for num_qubits
        if not (0 < boundary_qubit < self.master_circuit.num_qubits):
            raise ValueError(f"Boundary must be between 1 and {self.master_circuit.num_qubits - 1}.")

        # 1. Get the actual Circuit class and data
        CircuitClass = self.master_circuit.__class__
        labels = self.master_circuit.wire_labels

        # 2. Find the bridges (Run ONCE)
        self.bridges = _find_spanning_gates(self.master_circuit, boundary_qubit)
        spanning_indices = {idx for idx, _ in self.bridges}

        # 3. Create two new sub-circuits using master's class
        n_top = boundary_qubit
        n_bottom = self.master_circuit.num_qubits - boundary_qubit

        qc_top = CircuitClass(num_qubits=n_top, wire_labels=labels[:n_top])
        qc_bottom = CircuitClass(num_qubits=n_bottom, wire_labels=labels[n_top:])

        # 4. Route the gates safely using our optimized copy method
        for idx, op in enumerate(self.master_circuit.operations):
            if idx in spanning_indices:
                continue  # Skip the bridges! (They are stored in self.bridges)

            # Extract target indices
            indices = [t if isinstance(t, int) else t[1] for t in op["targets"]]

            if all(i < boundary_qubit for i in indices):
                # Purely Top: Add exactly as is
                qc_top.operations.extend(self.master_circuit._copy_ops([op]))

            elif all(i >= boundary_qubit for i in indices):
                # Purely Bottom: Copy and SHIFT the target indices!
                shifted_op = self.master_circuit._copy_ops([op])[0]
                new_targets = []
                for t in shifted_op["targets"]:
                    if isinstance(t, int):
                        new_targets.append(t - boundary_qubit)
                    else:
                        # Handles tuple formats like ('q', index)
                        new_targets.append((t[0], t[1] - boundary_qubit))

                shifted_op["targets"] = new_targets
                qc_bottom.operations.append(shifted_op)

        # Store them in the orchestrator
        self.sub_circuits["top"] = qc_top
        self.sub_circuits["bottom"] = qc_bottom

        return qc_top, qc_bottom, self.bridges

    def assign_simulator(self, sub_circuit_id: str, simulator):
        """Assigns a specific backend to a specific chunk."""
        pass

    def execute(self):
        """Executes the Pauli sum combinations across all sub-circuits."""
        pass
