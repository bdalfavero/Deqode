"""A CHP-style simulator for stabilizer circuits.
See https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.052328"""

from typing import List
import random
import numpy as np
import stim


class StabilizerTableau:

    def __init__(self, bmatrix):
        """Initializes a tableau from the binary matrix form."""

        if bmatrix.shape[0] % 2 == 0:
            raise ValueError("Matrix must have an odd number of rows!")

        if bmatrix.shape[1] != bmatrix.shape[0]:
            raise ValueError(
                f"Matrix has shape {bmatrix.shape}, but tableau should have shape (2 * n + 1) * (2 * n + 1)."
            )
        
        self._bmatrix = bmatrix
    
    @property
    def matrix(self):
        return self._bmatrix
    
    @property
    def nqubits(self):
        return (self._bmatrix.shape[0] - 1) // 2 
    
    def zero(nqubits: int):
        """Initialize the all-zero state."""

        bmatrix = np.zeros((2 * nqubits + 1, 2 * nqubits + 1), dtype=bool)
        for i in range(2 * nqubits):
            bmatrix[i, i] = True
        return StabilizerTableau(bmatrix)
    
    def to_stim_paulis(self) -> List[stim.PauliString]:
        """Convert the stabilizers into Stim-type Pauli strings."""

        raise NotImplementedError()
    
    def rowsum(self, h: int, i: int):

        def g(x1: bool, z1: bool, x2: bool, z2: bool) -> int:
            if (not x1) and (not z1):
                return 0
            elif x1 and z1:
                return int(z2) - int(x2)
            elif x1 and (not z1):
                return int(z2) * (2 * int(x2) - 1)
            else:
                return int(x2) * (1 - 2 * int(z2))
        
        # Set the sign bit conditioned on the current signs and the sum of g's.
        gsum = 0
        for j in range(self.nqubits):
            gsum += g(
                self._bmatrix[i, j], self._bmatrix[i, j + self.nqubits],
                self._bmatrix[h, j], self._bmatrix[h, j + self.nqubits]
            )
        ri = int(self._bmatrix[i, -1])
        rh = int(self._bmatrix[h, -1])
        self._bmatrix[h, -1] = (2 * ri + 2 * rh + gsum) % 4 != 0
        
        # Set x_hj and z_hj from XOR's.
        for j in range(self.nqubits):
            # x_hj <- x_ij XOR x_hj
            self._bmatrix[h, j] = self._bmatrix[i, j] ^ self._bmatrix[h, j]
            # z_hj <- z_ij XOR z_hj
            self._bmatrix[h, j + self.nqubits] = self._bmatrix[i, j + self.nqubits] \
                ^ self._bmatrix[h, j + self.nqubits]

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
    
    def measure(self, a: int) -> bool:
        """Measure qubit a."""

        assert a in range(0, self.nqubits)

        # Does there exist a p in [n+1, 2n] s.t. x_pa = 1?
        p_exists = False
        for p in range(self.nqubits, 2 * self.nqubits):
            if self._bmatrix[p, a]:
                p_exists = True
                break
        if p_exists:
            # Measurement result is random.
            for i in range(2 * self.nqubits):
                if i != p and self._bmatrix[i, a]:
                    self.rowsum(i, p)
            self._bmatrix[p - self.nqubits, :] = self._bmatrix[p, :].copy()
            self._bmatrix[p, :] = np.zeros(2 * self.nqubits + 1)
            r = random.random()
            self._bmatrix[p, -1] = r > 0.5
            self._bmatrix[p, a + self.nqubits] = True
            return self._bmatrix[p, -1]
        else:
            # Measurement result is deterministic.
            self._bmatrix[-1, :] = np.zeros(self._bmatrix.shape[1], dtype=bool)
            for i in range(self.nqubits):
                if self._bmatrix[i, a]:
                    self.rowsum(2 * self.nqubits, i + self.nqubits)
            return self._bmatrix[-1, -1]