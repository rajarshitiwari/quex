"""
Tests
-----

Tests for the package.
"""

# tests/test_circuit.py
import pytest
import numpy as np
import quex as qx


def test_circuit_initialization():
    """Test that a circuit initializes with the correct properties."""
    qc = qx.Circuit(num_qubits=3)

    assert qc.num_qubits == 3
    assert len(qc.wire_labels) == 3
    assert qc.num_gates == 0


def test_bell_state_probabilities():
    """Test the NumpySimulator using a standard Bell State."""
    qc = qx.Circuit(num_qubits=2)
    qc.add_operation("h", 0)
    qc.add_operation("cx", [0, 1])

    # Run exact statevector simulation
    qc.simulator = qx.NumpySimulator()
    qc.run()

    probs = qc.get_probabilities()

    # Bell state should have 50% chance of |00> and 50% chance of |11>
    assert len(probs) == 4
    assert np.isclose(probs[0], 0.5)  # |00>
    assert np.isclose(probs[3], 0.5)  # |11>
    assert np.isclose(probs[1], 0.0)  # |01>
    assert np.isclose(probs[2], 0.0)  # |10>


def test_hardware_sampling():
    """Test that shot sampling roughly matches exact probabilities."""
    qc = qx.Circuit(num_qubits=2)
    qc.add_operation("h", 0)
    qc.add_operation("cx", [0, 1])

    qc.simulator = qx.NumpySimulator()
    qc.run()

    shots = 10000
    counts = qc.sample_shots(shots=shots)

    # Check that both '00' and '11' are present and roughly 5000 each (allow 5% margin of error)
    assert "00" in counts
    assert "11" in counts
    assert 4500 < counts["00"] < 5500
    assert 4500 < counts["11"] < 5500


def test_fast_copy():
    """Test that the custom OpList and copy method create safe, independent circuits."""
    # Using your random ansatz generator from the notebook
    qc_original = qx.random_ansatz(num_qubits=3, depth=3)
    original_gate_count = qc_original.num_gates

    # Create the copy
    qc_copy = qc_original.copy()

    # Ensure they have the same data
    assert qc_copy.num_qubits == qc_original.num_qubits
    assert qc_copy.num_gates == original_gate_count

    # Ensure they are independent objects in memory
    assert qc_copy is not qc_original
    assert qc_copy.operations is not qc_original.operations

    # Ensure mutating the copy does NOT mutate the original (OpList cache safety)
    qc_copy.add_operation("x", 0)
    assert qc_copy.num_gates == original_gate_count + 1
    assert qc_original.num_gates == original_gate_count


def test_parameter_binding():
    """Test that parameterized circuits can be bound correctly."""
    qc = qx.Circuit(num_qubits=2)
    qc.add_operation("rx", 0, params=[np.pi])  # Bound
    qc.add_operation("ry", 1, params=["theta_1"])  # Free
    qc.add_operation("cx", [0, 1])

    # Check free parameters
    free_params = qc.free_parameters
    assert "theta_1" in free_params

    # Bind the parameter
    qc_bound = qc.bind_parameters({"theta_1": np.pi / 2})

    # Check the new circuit has no free parameters
    assert len(qc_bound.free_parameters) == 0
    # Original should remain unchanged
    assert "theta_1" in qc.free_parameters


def test_random_ansatz_U_generation():
    """Test that the massive U-gate ansatz generator does not crash."""
    try:
        # From your 12-qubit scratch test
        qc = qx.random_ansatz_U(12, depth=2)
        assert qc.num_qubits == 12
        assert qc.num_gates > 0
    except Exception as e:
        pytest.fail(f"random_ansatz_U failed to generate: {e}")
