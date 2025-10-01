import numpy as np
from deqode.chp_sim import StabilizerTableau
from deqode.gate import Gate

class Circuit:

    def __init__(self, nqubits: int):
        self._nqubits = nqubits
        self._tableau = StabilizerTableau.zero(nqubits)
        self._gates = []
    
    def __repr__(self):
        repr_str = ""
        for i, gate in enumerate(self._gates):
            repr_str += repr(gate)
            if i != len(self._gates) - 1:
                repr_str += "\n"
        return repr_str

    def append(self, gate: Gate):
        """Append a gate to the circuit."""

        self._gates.append(gate)
    
    def sample(self) -> np.ndarray:
        """Get a bitstring from the circuit."""

        results = []
        for gate in self._gates:
            if gate.is_measure:
                result = gate.apply_to(self._tableau)
                results.append(result)
            else:
                gate.apply_to(self._tableau)
        return np.array(results)