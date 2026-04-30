import concurrent.futures
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Prevent Cyclic Dependency
if TYPE_CHECKING:
    from quex.circuit import Circuit


class Simulator(ABC):
    """
    The abstract base prototype for all Quex computational engines.
    Behaves identically to an ASE 'Calculator'.
    """

    def __init__(self):
        self._circuit: Optional["Circuit"] = None

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, new_circuit: "Circuit"):
        self._circuit = new_circuit
        # Symmetric Link: Tell the circuit that WE are its simulator now.
        if new_circuit is not None and new_circuit.simulator is not self:
            new_circuit.simulator = self

    @abstractmethod
    def run(self, circuit: Optional["Circuit"] = None, parameter_binds: Dict[str, float] = None) -> Any:
        """Executes a single circuit."""
        pass

    def run_batch(self, circuits: List["Circuit"], parameter_binds_list: List[Dict[str, float]] = None) -> List[Any]:
        """
        Executes a list of circuits in parallel across all available CPU cores.
        """
        if parameter_binds_list is None:
            parameter_binds_list = [{}] * len(circuits)

        if len(circuits) != len(parameter_binds_list):
            raise ValueError("Length of circuits and parameter_binds_list must match.")

        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.run, circ, binds) for circ, binds in zip(circuits, parameter_binds_list)]

            for future in futures:
                results.append(future.result())

        return results
