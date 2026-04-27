"""
Visualisation module
--------------------

It contains functions and tools
to visualize circuits and other
relevant objects.

"""

import matplotlib.pyplot as plt
import networkx as nx


def draw_dag(parsed_operations):
    """
    Takes the parsed operation list from Quex and builds a DAG using NetworkX.
    """
    # 1. Initialize a Directed Graph
    dag = nx.DiGraph()

    # Keep track of the "last gate" applied to each qubit to draw the edges correctly
    # e.g., last_seen[('q', 0)] = "node_id_of_previous_gate"
    last_seen = {}

    # 2. Build the Graph
    for i, op in enumerate(parsed_operations):
        gate_name = op["gate"].upper()
        targets = op["targets"]

        # Create a unique node ID (e.g., "0: H", "1: CX")
        node_id = f"{i}: {gate_name}"
        dag.add_node(node_id, gate=gate_name, targets=targets)

        # Draw edges from the previous gate on these qubits to this new gate
        for qubit in targets:
            if qubit in last_seen:
                dag.add_edge(last_seen[qubit], node_id, label=f"q[{qubit[1]}]")

            # Update the last seen node for this qubit
            last_seen[qubit] = node_id

    # 3. Draw the Graph
    plt.figure(figsize=(8, 5))

    # Use a topological layout so time flows from left to right (or top to bottom)
    pos = nx.spring_layout(dag, seed=42)

    nx.draw(dag, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, font_weight="bold", arrows=True)

    # Add the edge labels (showing which qubit connects the gates)
    edge_labels = nx.get_edge_attributes(dag, "label")
    nx.draw_networkx_edge_labels(dag, pos, edge_labels=edge_labels, font_color="red")

    plt.title("Quex Internal Circuit DAG")
    plt.show()


def draw_structured_dag(dag):
    """Draws a NetworkX DAG organized left-to-right by time/depth."""
    plt.figure(figsize=(12, 6))  # Make it wider for timeline flow

    # 1. Group nodes into "generations" (layers of time)
    for layer, nodes in enumerate(nx.topological_generations(dag)):
        for node in nodes:
            dag.nodes[node]["layer"] = layer

    # 2. Use the multipartite layout to force left-to-right drawing
    pos = nx.multipartite_layout(dag, subset_key="layer", align="horizontal")

    # 3. Draw with smaller nodes and cleaner styling
    nx.draw(
        dag,
        pos,
        with_labels=True,
        node_color="#e0f2fe",  # Light modern blue
        edge_color="gray",
        node_size=800,
        font_size=8,
        arrows=True,
    )

    # Only draw edge labels (qubit wires) if it's not too cluttered
    if len(dag.nodes) < 50:
        edge_labels = nx.get_edge_attributes(dag, "label")
        nx.draw_networkx_edge_labels(dag, pos, edge_labels=edge_labels, font_size=7, font_color="red")

    plt.title("Quex Execution DAG (Time Flow ->)")
    plt.show()


def draw_qiskit(circuit):
    """
    Temporary dev-tool: Maps a Quex Circuit to a Qiskit QuantumCircuit
    and uses Qiskit's Matplotlib drawer to visualize it.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("Qiskit is not installed. To use the Qiskit visualizer, install dev dependencies: `uv add --dev qiskit matplotlib`")

    # 1. Initialize a Qiskit circuit with the exact same number of qubits
    qiskit_qc = QuantumCircuit(circuit.num_qubits)

    # 2. Iterate through your parsed operations and map them
    for op in circuit.operations:
        gate_name = op["gate"]
        params = op["params"]

        # Extract just the integer indices from your target tuples: e.g., ('q', 0) -> 0
        qubit_indices = [target[1] for target in op["targets"] if target[1] is not None]

        # Use Python's getattr to dynamically call the Qiskit function
        # e.g., if gate_name is 'h', this gets qiskit_qc.h
        gate_method = getattr(qiskit_qc, gate_name, None)

        if gate_method:
            if params:
                # E.g., qiskit_qc.rx(1.57, 0)
                gate_method(*params, *qubit_indices)
            else:
                # E.g., qiskit_qc.cx(0, 1)
                gate_method(*qubit_indices)
        else:
            print(f"Warning: Could not map Quex gate '{gate_name}' to Qiskit.")

    # 3. Draw the circuit using Matplotlib
    return qiskit_qc.draw(output="mpl", style="iqp")
