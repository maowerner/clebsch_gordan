"""Unit test for the utils functions
"""

import unittest
import numpy as np

import utils as ut

class TestCleanComplex(unittest.TestCase):
    def setUp(self):
        self.data = np.zeros((2,2), dtype=complex)
        self.addTypeEqualityFunc(np.ndarray, ut.check_array)

    def test_matrix_ones(self):
        self.data.fill(1.)
        res = ut.clean_complex(self.data)
        self.assertEqual(self.data, res)

    def test_matrix_imag_ones(self):
        self.data.fill(1.j)
        res = ut.clean_complex(self.data)
        self.assertEqual(self.data, res)

    def test_matrix_mixed_ones(self):
        self.data.fill(1.j+1.)
        res = ut.clean_complex(self.data)
        self.assertEqual(self.data, res)

    def test_special_matrix(self):
        self.data = np.asarray([[complex(0.5, -0.5), complex(-0.5, -0.5)],
                                [complex(0.5, -0.5), complex(0.5, 0.5)]])
        res = ut.clean_complex(self.data, prec=1e-6)
        self.assertEqual(self.data, res)

    def test_close_to_zero(self):
        self.data.fill(1e-20)
        res = ut.clean_complex(self.data)
        self.assertEqual(self.data, res)

class TestEq(unittest.TestCase):
    def test_same_vectors(self):
        vec = np.ones((3,))
        self.assertTrue(ut._eq(vec, vec))

    def test_different_vectors(self):
        vec = np.ones((3,))
        self.assertFalse(ut._eq(vec, 2*vec))

    def test_zero_vector(self):
        vec = np.zeros((3,))
        self.assertTrue(ut._eq(vec))

    def test_nonzero_vector(self):
        vec = np.zeros((3,))
        vec[1] += 1.
        self.assertFalse(ut._eq(vec))

    def test_high_precision(self):
        vec = np.zeros((3,))
        self.assertTrue(ut._eq(vec, prec=1e-20))

    def test_high_precision_nonzero(self):
        vec = np.zeros((3,)) + 1e-16
        self.assertFalse(ut._eq(vec, prec=1e-20))

    def test_same_vectors_complex(self):
        vec = np.ones((3,), dtype=complex) + 0.5j
        self.assertTrue(ut._eq(vec, vec))

    def test_different_vectors_complex(self):
        vec = np.ones((3,), dtype=complex) + 0.5j
        vec1 = np.ones((3,), dtype=complex)*0.5 + 2.j
        self.assertFalse(ut._eq(vec, vec1))

if __name__ == "__main__":
    unittest.main(verbosity=2)
