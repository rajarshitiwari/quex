"""
Cupy Simulator
--------------
"""

# src/quex/simulators/cupy_sim.py
from .numpy_sim import NumpySimulator


# Handle the optional dependency safely at the module level!
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


class CupySimulator(NumpySimulator):
    """
    GPU-accelerated statevector simulator using CuPy.
    Inherits the exact tensor contraction logic from NumpySimulator.
    """

    def __init__(self):
        # 1. Trigger the standard setup (which sets self._circuit = None)
        super().__init__()

        # 2. Overwrite the math backend for VRAM
        if not HAS_CUPY:
            raise ImportError("CuPy is not installed. Cannot use GPU backend.")

        self.xp = cp

        # 3. Create the dedicated VRAM cache to prevent PCIe bottlenecks
        self._gpu_tensor_cache = {}

    def _get_backend_tensor(self, name: str, params: list, num_targets: int):
        """
        Overrides the bridge! Fetches the CPU array from the parent,
        pushes it to the GPU, caches it, and returns the VRAM array.
        """
        # Create a unique, hashable key (lists aren't hashable, so convert params to tuple)
        cache_key = (name, tuple(params), num_targets)

        if cache_key not in self._gpu_tensor_cache:
            # 1. Ask the parent (NumpySimulator) for the standard np.ndarray
            cpu_tensor = super()._get_backend_tensor(name, params, num_targets)

            # 2. Push it across the PCIe bus to the VRAM and cache it!
            self._gpu_tensor_cache[cache_key] = self.xp.array(cpu_tensor)

        # 3. Return the VRAM array
        return self._gpu_tensor_cache[cache_key]

    def get_state(self, host: bool = True):
        """
        Safely retrieves the statevector.
        If host=True, downloads the array from GPU VRAM back to CPU RAM as a Numpy array.
        If host=False, returns the raw CuPy array for on-device hybrid workflows.
        """
        if self.state is None:
            return None

        if host:
            return self.xp.asnumpy(self.state)
        return self.state
