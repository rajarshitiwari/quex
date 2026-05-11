# src/quex/simulators/jax_sim.py
"""
Jax Simulator
-------
"""

from numpy.typing import ArrayLike

from quex.backends.numpy_sim import NumpySimulator

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


class JaxSimulator(NumpySimulator):
    """
    GPU/TPU-accelerated statevector simulator using Google JAX.
    Provides automatic differentiation capabilities.
    """

    def __init__(self):
        super().__init__()

        if not HAS_JAX:
            raise ImportError("JAX is not installed. Run `pip install jax jaxlib` (or `jax-metal` for Mac).")

        self.xp = jnp
        self.jax = jax
        self._gpu_tensor_cache = {}

    def _allocate_initial_state(self, num_qubits: int) -> ArrayLike:
        """
        Overrides allocation to use JAX's immutable syntax.
        state[(0,0...)] = 1.0 becomes state.at[(0,0...)].set(1.0)
        """
        state = self.xp.zeros((2,) * num_qubits, dtype=self.xp.complex128)

        # The JAX Way:
        state = state.at[(0,) * num_qubits].set(1.0)

        return state

    def _get_backend_tensor(self, name: str, params: list, num_targets: int):
        """Intercepts CPU Numpy arrays and pushes them to JAX DeviceArrays."""
        cache_key = (name, tuple(params), num_targets)

        if cache_key not in self._gpu_tensor_cache:
            cpu_tensor = super()._get_backend_tensor(name, params, num_targets)
            # Push to JAX device memory (GPU/TPU/Metal)
            self._gpu_tensor_cache[cache_key] = self.xp.array(cpu_tensor)

        return self._gpu_tensor_cache[cache_key]

    def get_state(self, host: bool = True):
        """Safely returns the state to standard Numpy if requested."""
        if self.state is None:
            return None

        if host:
            import numpy as np

            return np.array(self.state)  # jnp arrays convert cleanly back to np
        return self.state
