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
