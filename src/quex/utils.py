"""
Utility functions
-----------------

"""
import time
import numpy as np
import quex as qx

from collections import UserList
from typing import Generic, TypeVar

import numpy as np

# Generic Type variable
T = TypeVar("T")


class OpList(UserList, Generic[T]):
    """
    A list that triggers a callback whenever its contents are modified.
    Used to safely reset cached properties (like circuit layers) when
    the operations list is manipulated.
    """

    def __init__(self, callback, initial_data=None):
        super().__init__(initial_data)
        self._callback = callback

    # --- Mutating Methods Override ---
    def __setitem__(self, i, item):
        super().__setitem__(i, item)
        self._callback()

    def __delitem__(self, i):
        super().__delitem__(i)
        self._callback()

    def append(self, item):
        super().append(item)
        self._callback()

    def insert(self, i, item):
        super().insert(i, item)
        self._callback()

    def pop(self, i=-1):
        item = super().pop(i)
        self._callback()
        return item

    def extend(self, other):
        super().extend(other)
        self._callback()

    def remove(self, item):
        super().remove(item)
        self._callback()

    def clear(self):
        super().clear()
        self._callback()

    def __iadd__(self, other):
        """Catches the += operator"""
        super().__iadd__(other)
        self._callback()
        return self


def run_fused_benchmark(SimulatorClass, num_qubits=18, depth=100, max_fused=4):
    """
    Benchmarks sequential vs. fused execution for any given Simulator class.
    
    Returns:
        tuple: (seq_time, fused_time, is_accurate)
    """
    print(f"--- Benchmarking {SimulatorClass.__name__} | {num_qubits} Qubits, Depth {depth} ---")
    
    # 1. Generate the same circuit for both runs
    qc = qx.random_ansatz(num_qubits=num_qubits, depth=depth)
    
    # Safely bind parameters if the circuit has them
    parameters = getattr(qc, 'parameters', [])
    binds = {p: np.random.uniform(0, 2 * np.pi) for p in parameters}

    def _sync_and_get_state(simulator, state, statevector):
        """Helper to lock the clock for GPUs and pull memory back to CPU."""
        # JAX Synchronization
        if hasattr(state, 'block_until_ready'):
            state.block_until_ready()
        # CuPy Synchronization
        elif hasattr(simulator, 'xp') and simulator.xp.__name__ == 'cupy':
            import cupy as cp
            cp.cuda.Device().synchronize()
            
        # Bring memory back to standard numpy CPU array for comparison
        if hasattr(statevector, 'get'):
            return statevector.get()  # CuPy
        else:
            return np.array(statevector)  # NumPy / JAX

    # --- SEQUENTIAL MODE ---
    qc.simulator = SimulatorClass(exec_mode="sequential")
    start_time = time.time()
    qc.run(parameter_binds=binds)
    
    seq_state = _sync_and_get_state(qc.simulator, qc.state, qc.statevector)
    seq_time = time.time() - start_time
    
    print(f"Sequential Mode: {seq_time:.4f} seconds")

    # --- FUSED MODE ---
    qc.simulator = SimulatorClass(exec_mode="fused", max_fused_qubits=max_fused)
    start_time = time.time()
    qc.run(parameter_binds=binds)
    
    fused_state = _sync_and_get_state(qc.simulator, qc.state, qc.statevector)
    fused_time = time.time() - start_time

    print(f"Fused Mode (max {max_fused}): {fused_time:.4f} seconds")
    
    # --- ACCURACY CHECK ---
    is_accurate = np.allclose(seq_state, fused_state)
    print(f"Accuracy Match: {'✅ PASS' if is_accurate else '❌ FAIL'}")
    
    if seq_time > 0:
        speedup = seq_time / fused_time
        print(f"Speedup: {speedup:.2f}x\n")
        
    return seq_time, fused_time, is_accurate