"""A CHP-style simulator for stabilizer circuits.
See https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.052328"""

from typing import List
import numpy as np
import stim

class StabilizerTableau:

    def __init__(self, bmatrix):
        """Initializes a tableau from the binary matrix form."""

        if bmatrix.shape[0] % 2 != 0:
            raise ValueError("Matrix must have an even number of rows!")

        if bmatrix.shape[1] != bmatrix.shape[0] + 1:
            raise ValueError(
                f"Matrix has shape {bmatrix.shape}, but tableau should have shape (2 * n) * (2 * n + 1)."
            )
        
        self._bmatrix = bmatrix
    
    @property
    def matrix(self):
        return self._bmatrix
    
    @property
    def nqubits(self):
        return self._bmatrix.shape[0] // 2 
    
    def zero(nqubits: int):
        """Initialize the all-zero state."""

        bmatrix = np.zeros((2 * nqubits, 2 * nqubits + 1), dtype=bool)
        for i in range(2 * nqubits):
            bmatrix[i, i] = True
        return StabilizerTableau(bmatrix)
    
    def to_stim_paulis(self) -> List[stim.PauliString]:
        """Convert the stabilizers into Stim-type Pauli strings."""

        raise NotImplementedError()
    
    def h(self, a: int) -> None:
        """Apply a Hadamard gate to qubit a."""

        assert a in range(0, self.nqubits)

        for i in range(2 * self.nqubits):
            # r_i <- r_i oplus x_ia z_ia
            self._bmatrix[i, -1] = self._bmatrix[i, -1] ^ (self._bmatrix[i, a] * self._bmatrix[i, a+self.nqubits])
            # Swap x_ia and z_ia
            temp = self._bmatrix[i, a] # x_ia
            self._bmatrix[i, a] = self._bmatrix[i, a+self.nqubits]
            self._bmatrix[i, a+self.nqubits] = temp

    def phase(self, a: int) -> None:
        """Apply an S gate to qubit a."""

        assert a in range(0, self.nqubits)

        for i in range(2 * self.nqubits):
            # r_i <- r_i oplus x_ia z_ia
            self._bmatrix[i, -1] = self._bmatrix[i, -1] ^ (self._bmatrix[i, a] * self._bmatrix[i, a+self.nqubits])
            # z_ia <- z_ia oplus x_ia
            self._matrix[i, a+self.nqubits] = self._matrix[i, a] ^ self._matrix[i, a+self.nqubits]
    
    def cnot(self, a: int, b: int) -> None:
        """Apple a CNOT gate controlled by qubit a and acting on qubit b."""

        assert a in range(0, self.nqubits)
        assert b in range(0, self.nqubits)
        assert a != b

        for i in range(2 * self.nqubits):
            x_ia = self._bmatrix[i, a]
            x_ib = self._bmatrix[i, b]
            z_ia = self._bmatrix[i, a+self.nqubits]
            z_ib = self._bmatrix[i, b+self.nqubits]
            # r_i <- r_i oplus x_ia z_ib (x_ib oplus z_ia oplus 1)
            self._bmatrix[i, -1] = self._bmatrix[i, -1] ^ x_ia & z_ib & (x_ib ^ z_ia ^ 1)
            # x_ib <- x_ib oplus x_ia
            self._bmatrix[i, b] = x_ib ^ x_ia
            # z_ia <- z_ia oplus z_ib
            self._bmatrix[i, a+self.nqubits] = z_ia ^ z_ib