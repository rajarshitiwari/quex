# src/quex/circuit.py

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

import numpy as np

from quex.parser import parse_qasm_string
from quex.utils import OpList

if TYPE_CHECKING:
    from quex.backends.base import Simulator


class Circuit:
    """
    The internal, backend-agnostic representation of a quantum circuit in Quex.
    """

    def __init__(self, num_qubits: int, wire_labels: Optional[List[str]] = None):
        """
        The simplest state: Initialize a blank circuit with N qubits.
        """
        self.num_qubits = num_qubits

        self._layers = None
        # New: The Attached Simulator (Calculator equivalent)
        self._simulator: Optional["Simulator"] = None

        # The flat list of operations (Gate Name, Params, Targets)
        self.operations: OpList[Dict[str, Any]] = OpList(callback=self._reset_cache)

        # New: a dictionary for parameters.
        self.parameters: Dict[str, float] = {}
        # If  no labels provided, default to q[0]..q[n] ...
        if wire_labels is None:
            self.wire_labels = [f"q[{i}]" for i in range(num_qubits)]
        else:
            self.wire_labels = wire_labels

    def _reset_cache(self):
        """Wipes cached properties so they rebuild on next access."""
        self._layers = None
        # If we put anything similar like self._depth_cache, we just add them here!

    def __add__(self, other: "Circuit") -> "Circuit":
        """
        Horizontal Concatenation (Sequence): qc1 + qc2
        Appends qc2 to the end of qc1. Both must have the same number of qubits.
        """
        if not isinstance(other, Circuit):
            return NotImplemented
        if self.num_qubits != other.num_qubits:
            raise ValueError(f"Cannot add circuits with different qubit counts ({self.num_qubits} vs {other.num_qubits}).")

        # Create a new blank circuit
        new_qc = Circuit(self.num_qubits, wire_labels=self.wire_labels.copy())

        # Merge operations (Deep copy to prevent linking memory)
        new_qc.operations = self._copy_ops(self.operations) + self._copy_ops(other.operations)

        # Merge parameters (other overwrites self if there are naming collisions)
        new_qc.parameters = {**self.parameters, **other.parameters}

        return new_qc

    def __and__(self, other: "Circuit") -> "Circuit":
        """
        Vertical Concatenation (Tensor Product): qc1 & qc2
        Stacks qc2 'below' qc1, expanding the total number of qubits.
        """
        if not isinstance(other, Circuit):
            return NotImplemented

        total_qubits = self.num_qubits + other.num_qubits
        new_labels = self.wire_labels + other.wire_labels
        new_qc = Circuit(total_qubits, wire_labels=new_labels)

        # 1. Add operations from the top circuit (indices stay the same)
        new_qc.operations = self._copy_ops(self.operations)

        # 2. Add operations from the bottom circuit (Shift their target indices!)
        shift = self.num_qubits
        for op in self._copy_ops(other.operations):
            shifted_targets = []
            for target_type, target_idx in op["targets"]:
                shifted_targets.append((target_type, target_idx + shift))
            op["targets"] = shifted_targets
            new_qc.operations.append(op)

        # 3. Merge parameters
        new_qc.parameters = {**self.parameters, **other.parameters}

        return new_qc

    def __mul__(self, repetitions: int) -> "Circuit":
        """
        Repetition: qc * 3
        Repeats the circuit 'n' times sequentially.
        qc * 0 returns an empty Identity circuit on the same qubits.
        """
        if not isinstance(repetitions, int) or repetitions < 0:
            raise ValueError("Circuit multiplication requires a positive integer.")

        new_qc = Circuit(self.num_qubits, wire_labels=self.wire_labels.copy())
        new_qc.parameters = self.parameters.copy()

        for _ in range(repetitions):
            new_qc.operations.extend(self._copy_ops(self.operations))

        return new_qc

    def __rmul__(self, repetitions: int) -> "Circuit":
        """
        Right-side multiplication: 3 * qc
        Routes perfectly back to standard multiplication.
        """
        return self.__mul__(repetitions)

    @staticmethod
    def _copy_ops(ops_list: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        shallow-deep copy for the operations schema.
        Bypasses the overhead of Python's generic copy.deepcopy().
        """
        return [
            {
                "gate": op["gate"],  # Strings are immutable, safe to reference
                "targets": op["targets"].copy() if isinstance(op["targets"], list) else op["targets"],
                "params": op["params"].copy() if isinstance(op["params"], list) else op["params"],
            }
            for op in ops_list
        ]

    @property
    def wire_labels(self) -> List[str]:
        return self._wire_labels

    @wire_labels.setter
    def wire_labels(self, labels: List[str]):
        """Safely updates labels, ensuring the length matches the qubit count."""
        if len(labels) != self.num_qubits:
            raise ValueError(f"Provided {len(labels)} labels, but circuit has {self.num_qubits} qubits.")
        self._wire_labels = labels

    def reset_labels(self, prefix: str = "q"):
        """
        Resets all wire labels to a continuous sequential default.
        Example: qc.reset_labels("sys") -> sys[0], sys[1], ...
        """
        self.wire_labels = [f"{prefix}[{i}]" for i in range(self.num_qubits)]

    # --- NEW: Flattened view on demand ---
    @property
    def statevector(self) -> Optional[np.ndarray]:
        """Returns the 1D flattened view of the N-dimensional state tensor."""
        if self.state is None:
            return None
        return self.state.flatten()

    @property
    def simulator(self):
        return self._simulator

    @simulator.setter
    def simulator(self, calc: "Simulator"):
        self._simulator = calc
        # Symmetric Link: Tell the simulator that WE are its circuit now.
        if calc is not None and calc.circuit is not self:
            calc.circuit = self

    def copy(self):
        """Returns an independent copy of the circuit."""
        new_qc = self.__class__(self.num_qubits, self.wire_labels.copy())
        copied_ops = self._copy_ops(self.operations)
        new_qc.operations.extend(copied_ops)
        # new_qc.operations = self._copy_ops(self.operations)
        return new_qc

    def pop(self, index: int = -1) -> dict:
        """
        Removes and returns an operation. Defaults to the last gate.
        Old self.layer gets reset now, as circuit changed!
        """
        popped_op = self.operations.pop(index)
        return popped_op

    def insert(self, index: int, gate: str, targets: list, params: list = None):
        """
        Inserts a gate at a specific index in the execution timeline.
        Old self.layer gets reset now, as circuit changed!
        """
        op = {"gate": gate, "targets": targets if isinstance(targets, list) else [targets], "params": params or []}
        self.operations.insert(index, op)

    # --- UPDATED: Allow passing an initial state for composition! ---
    def run(self, parameter_binds: dict = None, initial_state: np.ndarray = None):
        """Delegates simulation to the attached Simulator."""
        if self._simulator is None:
            raise RuntimeError("No Simulator attached! Assign qc.simulator = NumpySimulator() first.")
        return self._simulator.run(circuit=self, parameter_binds=parameter_binds, initial_state=initial_state)

    def get_probabilities(self) -> np.ndarray:
        """
        Calculates the exact mathematical probability of measuring each basis state.
        Returns a 1D array of probabilities (P = |psi|^2).
        """
        if self.state is None:
            raise RuntimeError("Circuit has no state. Please call qc.run() first.")

        probs = np.abs(self.state) ** 2
        return probs.flatten()

    def sample_shots(self, shots: int = 1024) -> dict:
        """
        Simulates quantum hardware by collapsing the statevector 'shots' times.
        Returns a dictionary of measured bitstrings and their counts.
        """
        if self.state is None:
            raise RuntimeError("Circuit has no state. Please call qc.run() first.")

        probs = self.get_probabilities()

        # Generate all possible bitstrings for this number of qubits
        format_str = f"{{:0{self.num_qubits}b}}"
        bitstrings = [format_str.format(i) for i in range(2**self.num_qubits)]

        # Roll the quantum dice!
        samples = np.random.choice(bitstrings, size=shots, p=probs)

        # Tally the results
        counts = {}
        for bitstring in samples:
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return dict(sorted(counts.items()))

    def add_operation(self, gate: str, targets: Union[int, List[int], List[tuple]], params: Optional[List[Union[float, str]]] = None):
        """
        Programmatically add an operation to the circuit.
        Friendly UX: 'targets' accepts a single int, a list of ints, or internal tuples.
        Automatically registers any new string parameters in the circuit's state.
        """
        if params is None:
            params = []

        # --- NEW: Parameter Auto-Registration (ASE-style state tracking) ---
        for p in params:
            # If the parameter is a variable name (string) we haven't seen before,
            # auto-populate it in the circuit's state with a safe 0.0 default.
            if isinstance(p, str) and p not in self.parameters:
                self.parameters[p] = 0.0
        # -------------------------------------------------------------------

        # --- Input Normalisation ---
        normalised_targets = []

        # Case 1: User passed a single integer -> qc.add_operation('x', 0)
        if isinstance(targets, int):
            normalised_targets.append(("q", targets))

        # Case 2: User passed a list
        elif isinstance(targets, list):
            for t in targets:
                if isinstance(t, int):
                    # qc.add_operation('cx', [0, 1])
                    normalised_targets.append(("q", t))
                elif isinstance(t, tuple):
                    # Internal parser format: qc.add_operation('x', [('q', 0)])
                    normalised_targets.append(t)
                else:
                    raise ValueError(f"Target {t} must be an integer or tuple.")
        else:
            raise ValueError("Targets must be an int, list of ints, or list of tuples.")

        op = {"gate": gate.lower(), "params": params, "targets": normalised_targets}
        self.operations.append(op)

        # --- if circuit changed, re-evaluate layers ---
        self._layers = None

    @property
    def layers(self) -> List[List[Dict[str, Any]]]:
        """
        Returns the operations grouped by parallel execution moments.
        Calculates only when necessary (Lazy Evaluation).
        """
        if self._layers is None:
            self._layers = self._build_layers()
        return self._layers

    @property
    def num_gates(self) -> int:
        """Returns the total number of operations in the circuit."""
        return len(self.operations)

    @property
    def depth(self) -> int:
        """
        Circuit depth computation

        Returns
        -------
        int
            Circuit depth
        """
        return len(self.layers)

    @property
    def free_parameters(self) -> list:
        """Returns a sorted list of all unbound parameter names (strings) in the circuit."""
        params = set()
        for op in self.operations:
            if op["params"]:
                for p in op["params"]:
                    if isinstance(p, str):
                        params.add(p)
        return sorted(list(params))

    def bind_parameters(self, binds: dict) -> "Circuit":
        """
        Returns a NEW Circuit instance with the specified parameters bound.
        Supports partial binding (leaving some parameters as strings).
        """
        # Create a fresh circuit with the exact same geometry
        new_qc = self.__class__(num_qubits=self.num_qubits, wire_labels=self.wire_labels)

        for op in self.operations:
            new_params = []
            if op["params"]:
                for p in op["params"]:
                    # If it is a variable AND the user provided a value for it, swap it!
                    if isinstance(p, str) and p in binds:
                        new_params.append(float(binds[p]))
                    else:
                        # Keep it as is (either already a float, or a still-unbound string)
                        new_params.append(p)

            # Safely add the operation to the new circuit
            new_qc.add_operation(gate=op["gate"], targets=op["targets"], params=new_params)
        return new_qc

    def _build_layers(self) -> list:
        """
        Organizes the flat list of operations into parallel execution layers (moments).
        This determines the true 'depth' of the circuit and enables parallel batched execution.
        """
        wire_depths = {i: 0 for i in range(self.num_qubits)}
        layers = []

        for op in self.operations:
            targets = [t[1] for t in op["targets"] if t[1] is not None]
            if not targets:
                continue

            # A multi-qubit gate blocks all wires between its top and bottom targets
            top, bottom = min(targets), max(targets)
            blocked_wires = range(top, bottom + 1)

            # Find the deepest wire this gate touches
            op_depth = max(wire_depths[w] for w in blocked_wires)

            # Create a new layer if needed
            while len(layers) <= op_depth:
                layers.append([])

            layers[op_depth].append(op)

            # Update the depth tracker for all blocked wires
            for w in blocked_wires:
                wire_depths[w] = op_depth + 1

        return layers

    @classmethod
    def from_qasm(cls, qasm_string: str) -> "Circuit":
        """Builds a Quex Circuit directly from an OpenQASM 3 string."""
        parsed_data = parse_qasm_string(qasm_string)
        registers = parsed_data["registers"]
        parsed_ops = parsed_data["operations"]

        # 1. Build the Virtual-to-Physical Map
        # If registers = {'q': 3, 'qq': 2}, mapping becomes {'q': 0, 'qq': 3}
        reg_starting_wire = {}
        wire_labels = []  # Store the exact labels
        current_wire = 0

        for reg_name, size in registers.items():
            reg_starting_wire[reg_name] = current_wire
            for i in range(size):
                wire_labels.append(f"{reg_name}[{i}]")
            current_wire += size

        total_qubits = current_wire
        # Now pass the labels into the Circuit!
        circuit = cls(num_qubits=total_qubits, wire_labels=wire_labels)

        # 2. Translate the operations to use flat physical wires
        for op in parsed_ops:
            flat_targets = []
            for reg_name, reg_idx in op["targets"]:
                # The math: Starting wire + offset index
                physical_wire = reg_starting_wire[reg_name] + reg_idx

                # We save it as ("q", physical_wire) so our visualizer doesn't break!
                flat_targets.append(("q", physical_wire))

            circuit.add_operation(gate=op["gate"], targets=flat_targets, params=op["params"])

        return circuit

    def to_text_diagram(self) -> str:
        """
        Generates a native ASCII/Unicode representation of the quantum circuit,
        visually compressing parallel operations into aligned vertical columns.
        """
        if self.num_qubits == 0:
            return "Empty Circuit"

        # get the optimised layers of operation
        layers = self.layers

        # --- 2. SETUP WIRES ---
        max_label_len = max(len(label) for label in self.wire_labels)
        wires = [f"{self.wire_labels[i].rjust(max_label_len)}: ──" for i in range(self.num_qubits)]
        prefix_len = max_label_len + 4
        spacers = [" " * prefix_len for _ in range(self.num_qubits - 1)]

        # --- 3. DRAW COLUMN BY COLUMN ---
        for layer in layers:
            # Find the widest gate in THIS specific layer for alignment
            col_width = 4
            for op in layer:
                gate_name = op["gate"].upper()
                gate_str = gate_name
                if op["params"]:
                    params_str = ",".join(str(round(p, 2)) if isinstance(p, (int, float)) else str(p) for p in op["params"])
                    gate_str = f"{gate_name}({params_str})"
                col_width = max(col_width, len(gate_str) + 4)

            # Blank slates for this specific column
            wire_chars = ["─" * col_width for _ in range(self.num_qubits)]
            spacer_chars = [" " * col_width for _ in range(self.num_qubits - 1)]

            for op in layer:
                gate_name = op["gate"].upper()
                gate_str = gate_name
                if op["params"]:
                    params_str = ",".join(str(round(p, 2)) if isinstance(p, (int, float)) else str(p) for p in op["params"])
                    gate_str = f"{gate_name}({params_str})"

                targets = [t[1] for t in op["targets"] if t[1] is not None]

                if len(targets) == 1:
                    t = targets[0]
                    wire_chars[t] = f"[{gate_str}]".center(col_width, "─")

                elif len(targets) >= 2:
                    tgt = targets[-1]
                    controls = targets[:-1]
                    top, bottom = min(targets), max(targets)

                    for i in range(top, bottom + 1):
                        if i in controls:
                            symbol = "X" if gate_name == "SWAP" else "■"
                            wire_chars[i] = symbol.center(col_width, "─")
                        elif i == tgt:
                            if gate_name in ["CX", "CCX", "TOFFOLI", "SWAP"]:
                                symbol = "X"
                            elif gate_name in ["CZ", "CCZ"]:
                                symbol = "■"
                            else:
                                symbol = f"[{gate_str}]"
                            wire_chars[i] = symbol.center(col_width, "─")
                        else:
                            wire_chars[i] = "│".center(col_width, "─")

                    # Fill the spacers with pipes between the bounds
                    for i in range(top, bottom):
                        spacer_chars[i] = "│".center(col_width, " ")

            # Append the rendered column to the main drawing
            for i in range(self.num_qubits):
                wires[i] += wire_chars[i]
                if i < self.num_qubits - 1:
                    spacers[i] += spacer_chars[i]

        # --- 4. INTERLEAVE AND RETURN ---
        output = []
        for i in range(self.num_qubits):
            output.append(wires[i])
            if i < self.num_qubits - 1:
                output.append(spacers[i])

        return "\n".join(output)

    def __repr__(self) -> str:
        """Allows the circuit to draw itself automatically in standard REPLs."""
        return self.to_text_diagram()

    def __str__(self) -> str:
        """Allows `print(circuit)` to output the text diagram."""
        return self.to_text_diagram()

    def to_qasm(self) -> str:
        """
        Generates a strictly valid OpenQASM 3.0 string representation of the circuit.
        """
        # Standard OpenQASM 3 headers
        lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', ""]

        # 1. Reconstruct Register Declarations from wire_labels
        # Scans labels like 'q[0]', 'q[1]', 'qq[0]' to build {'q': 2, 'qq': 1}
        registers = {}
        for label in self.wire_labels:
            if "[" in label and label.endswith("]"):
                reg_name, idx_str = label.split("[")
                idx = int(idx_str[:-1])
                # Track the maximum index to determine the register size
                registers[reg_name] = max(registers.get(reg_name, 0), idx + 1)
            else:
                # Fallback if a user provided a custom label without brackets
                registers[label] = 1

        for reg_name, size in registers.items():
            if size > 1:
                lines.append(f"qubit[{size}] {reg_name};")
            else:
                lines.append(f"qubit {reg_name};")

        # --- NEW: Declare Free Parameters for Strict OpenQASM 3.0 Validity ---
        free_params = self.free_parameters
        if free_params:
            lines.append("")
            for param in free_params:
                lines.append(f"input angle {param};")
        # ---------------------------------------------------------------------

        lines.append("")

        # 2. Reconstruct Operations sequentially
        for op in self.operations:
            gate = op["gate"]

            # Format parameters (Seamlessly handles both floats and Late-Bound Strings!)
            if op["params"]:
                params_str = ", ".join(str(p) for p in op["params"])
                gate_part = f"{gate}({params_str})"
            else:
                gate_part = gate

            # Map the physical indices back to their exact string labels (e.g., 'qq[0]')
            target_labels = [self.wire_labels[t[1]] for t in op["targets"] if t[1] is not None]
            targets_str = ", ".join(target_labels)

            lines.append(f"{gate_part} {targets_str};")

        return "\n".join(lines)

    def _find_spanning_gates(self, boundary_qubit: int) -> list:
        """
        Finds all 2+ qubit gates that cross the boundary.
        The boundary divides the circuit into:
        Top: qubits 0 to boundary_qubit - 1
        Bottom: qubits boundary_qubit to N - 1

        Returns a list of tuples: (operation_index, operation_dict)
        """
        spanning_gates = []

        for idx, op in enumerate(self.operations):
            # We only care about gates with multiple targets
            if isinstance(op["targets"], list) and len(op["targets"]) > 1:
                # Extract just the integer indices from the targets list
                # Assuming targets look like: [('q', 0), ('q', 1)] or just [0, 1]
                # (Adjust this depending on how you strictly store targets internally!)
                indices = [t if isinstance(t, int) else t[1] for t in op["targets"]]

                has_top = any(i < boundary_qubit for i in indices)
                has_bottom = any(i >= boundary_qubit for i in indices)

                if has_top and has_bottom:
                    spanning_gates.append((idx, op))

        return spanning_gates

    def cleave(self, boundary_qubit: int):
        """
        Cleaves the circuit horizontally into two independent sub-circuits.
        Top circuit: qubits 0 to boundary_qubit - 1
        Bottom circuit: qubits boundary_qubit to N - 1

        Returns:
            qc_top (Circuit): The upper sub-circuit.
            qc_bottom (Circuit): The lower sub-circuit.
            spanning_gates (list): The metadata of the gates that were cut.
        """
        if not (0 < boundary_qubit < self.num_qubits):
            raise ValueError(f"Boundary must be between 1 and {self.num_qubits - 1}.")

        # 1. Initialize the two independent sub-circuits
        # We use self.__class__ to instantiate new circuits safely
        qc_top = self.__class__(num_qubits=boundary_qubit, wire_labels=self.wire_labels[:boundary_qubit].copy())
        qc_bottom = self.__class__(num_qubits=self.num_qubits - boundary_qubit, wire_labels=self.wire_labels[boundary_qubit:].copy())

        # 2. Find the bridges
        spanning_gates_info = self._find_spanning_gates(boundary_qubit)
        spanning_indices = {idx for idx, _ in spanning_gates_info}

        # 3. Route the gates safely using our optimized copy method
        for idx, op in enumerate(self.operations):
            if idx in spanning_indices:
                continue  # Skip the bridges! (They are returned as metadata)

            # Extract target indices
            indices = [t if isinstance(t, int) else t[1] for t in op["targets"]]

            if all(i < boundary_qubit for i in indices):
                # Purely Top: Add exactly as is
                qc_top.operations.extend(self._copy_ops([op]))

            elif all(i >= boundary_qubit for i in indices):
                # Purely Bottom: Copy and SHIFT the target indices!
                shifted_op = self._copy_ops([op])[0]
                new_targets = []
                for t in shifted_op["targets"]:
                    if isinstance(t, int):
                        new_targets.append(t - boundary_qubit)
                    else:
                        # Handles tuple formats like ('q', index)
                        new_targets.append((t[0], t[1] - boundary_qubit))

                shifted_op["targets"] = new_targets
                qc_bottom.operations.append(shifted_op)

        return qc_top, qc_bottom, spanning_gates_info


def reconstruct_single_cut(qc_original: Circuit, boundary_qubit: int, bridge_info: tuple, simulator=None):
    """
    Reconstructs the full statevector after cutting a single CX gate,
    preserving the exact temporal order of operations.
    """
    # Resolve which simulator backend to use: the passed simulator,
    # or inherit the one attached to the original circuit
    sim_backend = simulator or qc_original._simulator
    if sim_backend is None:
        raise RuntimeError("No simulator found. Attach a simulator to qc_original or pass one as an argument.")

    idx, op = bridge_info

    # The CX Decomposition: 0.5 * (II + ZI + IX - ZX)
    coefficients = [0.5, 0.5, 0.5, -0.5]
    paulis_top = ["id", "z", "id", "z"]
    paulis_bottom = ["id", "id", "x", "x"]

    # The final 4-qubit statevector starts completely empty
    total_state = np.zeros(2 ** (qc_original.num_qubits), dtype=np.complex128)

    print("\n--- RUNNING SUB-CIRCUITS ---")
    # Run the 4 parallel universes
    for i in range(4):
        print(f"Term {i + 1}: Applying {paulis_top[i]} (Top) and {paulis_bottom[i]} (Bottom)")

        # 1. Create a fresh copy of the ORIGINAL master circuit
        temp_master = qc_original.copy()

        # 2. Time Travel: Remove the entangling bridge
        temp_master.operations.pop(idx)

        # 3. Insert the local Paulis exactly where the bridge used to be!
        temp_master.insert(index=idx, gate=paulis_bottom[i], targets=[op["targets"][1]])
        temp_master.insert(index=idx, gate=paulis_top[i], targets=[op["targets"][0]])

        # 4. Cleave this newly un-entangled master circuit
        qc_top, qc_bottom, _ = temp_master.cleave(boundary_qubit)

        # 5. inject the resolved simulator backend
        qc_top.simulator = sim_backend
        qc_bottom.simulator = sim_backend

        qc_top.run()
        qc_bottom.run()

        # 6. The Magic: Flatten and Kronecker product
        state_top_1d = qc_top.state.flatten()
        state_bot_1d = qc_bottom.state.flatten()

        combined_term = coefficients[i] * np.kron(state_top_1d, state_bot_1d)
        total_state += combined_term

    return total_state
