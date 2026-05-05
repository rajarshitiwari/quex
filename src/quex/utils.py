"""
Utility functions
-----------------

"""

import copy

import numpy as np

import quex as qx


def reconstruct_single_cut(qc_original: qx.Circuit, boundary_qubit: int, bridge_info: tuple):
    """
    Reconstructs the full statevector after cutting a single CX gate,
    preserving the exact temporal order of operations.
    """
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
        temp_master = copy.deepcopy(qc_original)

        # 2. Time Travel: Remove the entangling bridge
        temp_master.operations.pop(idx)

        # 3. Insert the local Paulis exactly where the bridge used to be!
        # (Using lists for targets to keep the schema consistent)
        op_bottom = {"gate": paulis_bottom[i], "targets": [op["targets"][1]], "params": []}
        op_top = {"gate": paulis_top[i], "targets": [op["targets"][0]], "params": []}

        temp_master.operations.insert(idx, op_bottom)
        temp_master.operations.insert(idx, op_top)

        # 4. Cleave this newly un-entangled master circuit
        qc_top, qc_bottom, _ = temp_master.cleave(boundary_qubit)

        # 5. Run the small, fast local simulators
        qc_top.simulator = qx.NumpySimulator()
        qc_bottom.simulator = qx.NumpySimulator()
        qc_top.run()
        qc_bottom.run()

        # 6. The Magic: Flatten and Kronecker product
        state_top_1d = qc_top.state.flatten()
        state_bot_1d = qc_bottom.state.flatten()

        combined_term = coefficients[i] * np.kron(state_top_1d, state_bot_1d)
        total_state += combined_term

    return total_state
