"""Unit test for the utils functions
"""

import unittest
import numpy as np

import utils as ut

class TestCleanComplex(unittest.TestCase):
    def setUp(self):
        self.data = np.zeros((2,2), dtype=complex)
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

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

if __name__ == "__main__":
    unittest.main(verbosity=2)