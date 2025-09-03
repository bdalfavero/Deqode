import unittest
import numpy as np
from deqode.chp_sim import StabilizerTableau

class TestHadamard(unittest.TestCase):

    def test_hh_on_00(self):
        """If we apply HH to the state |00>, the resulting state should be
        stabilized by XI and IX."""

        bmatrix = np.array([
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, True, False, False],
            [False, False, False, True, False]
        ])
        tableau = StabilizerTableau(bmatrix)
        tableau.h(0)
        tableau.h(1)
        target_bmatrix = np.array([
            [False, False, True, False, False],
            [False, False, False, True, False],
            [True, False, False, False, False],
            [False, True, False, False, False]
        ])
        self.assertTrue(np.all(target_bmatrix == tableau.matrix))

    def test_hh_on_bell(self):
        """If we apply HH to the state a Bell state, we should get a computational basis state.
        If we use 1/sqrt(2) (|00> + |11>), then H_1 H_2 should give us the same state back."""

        bmatrix = np.array([
            [True, False, False, False, False],
            [False, False, True, False, False],
            [False, False, True, True, False],
            [True, True, False, False, False]
        ])
        tableau = StabilizerTableau(bmatrix)
        tableau.h(0)
        tableau.h(1)
        target_bmatrix = np.array([
            [False, False, True, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
            [False, False, True, True, False]
        ])
        self.assertTrue(np.all(target_bmatrix == tableau.matrix))
    
    def test_h_squared_is_id(self):
        """If we apply H twice to the same qubit, the state should remain the same."""

        nq = 3
        tableau = StabilizerTableau.zero(nq)
        old_matrix = tableau.matrix.copy()
        for i in range(nq):
            tableau.h(i)
            tableau.h(i)
        assert np.all(tableau.matrix == old_matrix)


class TestCNOT(unittest.TestCase):

    def test_cnot_twice_is_id(self):
        # TODO This test seems to produce the same group, but not the same tableaus.
        # Prepare a maximally-mixed state.
        nq = 4
        tableau = StabilizerTableau.zero(nq)
        for i in range(nq):
            tableau.h(i)
        old_matrix = tableau.matrix.copy()
        print(old_matrix)
        # Apply two CNOTs on neighboring qubits.
        for i in range(nq - 1):
            tableau.cnot(i, i+1)
        print(tableau.matrix)
        self.assertTrue(np.all(tableau.matrix == old_matrix))


if __name__ == "__main__":
    unittest.main()