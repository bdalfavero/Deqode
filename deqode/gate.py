from typing import List
import numpy as np
from deqode.chp_sim import StabilizerTableau

class Gate:
    """A operation (either unitary or not) applied to a qubit."""

    def __init__(self, name: str, targets: List[int], is_noisy: bool, is_measure: bool):
        self._name = name
        self._targets = targets
        self._is_noisy = is_noisy
        self._is_measure = is_measure
    
    @property
    def targets(self):
        return self._targets
    
    @property
    def is_noisy(self):
        return self._is_noisy
    
    @property
    def is_measure(self):
        return self._is_measure
    
    def __repr__(self):
        return self._name + ' ' + ' '.join([str(t) for t in self._targets])

    def apply_to(self, tableau: StabilizerTableau):
        """Apply the gate to the tableau in place."""

        raise NotImplemented("This class does not impelement apply.")


class HadamardGate(Gate):

    def __init__(self, target: int):
        self._name = "H"
        self._targets = [target]
        self._is_noisy = False
        self._is_measure = False
    
    def apply_to(self, tableau: StabilizerTableau):
        tableau.h(self._targets[0])


class CNOTGate(Gate):

    def __init__(self, ctrl: int, target: int):
        self._name = "CX"
        self._targets = [ctrl, target]
        self._is_noisy = False
        self._is_measure = False
    
    def apply_to(self, tableau: StabilizerTableau):
        tableau.cnot(self._targets[0], self._targets[1])


class MeasureGate(Gate):

    def __init__(self, target: int):
        self._name = "M"
        self._targets = [target]
        self._is_noisy = False
        self._is_measure = True

    def apply_to(self, tableau: StabilizerTableau) -> bool:
        return tableau.measure(self._targets[0])