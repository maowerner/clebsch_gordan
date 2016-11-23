"""Unit test for the group class
"""

import unittest
import numpy as np

import utils
import group_class as gc
import group_basis as gb

class TestBasis(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)
        self.g = gc.OhGroup(instances=True)
        self.b = gb.BasisIrrep(1, self.g)

    def test_attributes(self):
        self.assertEqual(self.b.p2, 0)
        self.assertEqual(self.b.jmax, 1)
        self.assertEqual(self.b.lm, [(0,0)])
        self.assertEqual(self.b.IR[0], ("A1",0))
        self.assertEqual(self.b.dims, (1, 18, 1))

    def test_basis(self):
        tmp = np.zeros((1, 18, 1), dtype=complex)
        tmp[0,0,0] = 1.
        self.assertEqual(self.b.basis, tmp)

    def test_coefficient(self):
        res = self.b.coefficient("A1", 0, 0, 0)
        self.assertEqual(res, 1.)
        res = self.b.coefficient("T1", 0, 0, 0)
        self.assertEqual(res, 0.)
        res = self.b.coefficient("T1", 1, 0, 0)
        self.assertEqual(res, 0.)
        res = self.b.coefficient("T1", 2, 0, 0)
        self.assertEqual(res, 0.)

    def test_coefficient_wrong_irrep(self):
        res = self.b.coefficient("K1", 0, 0, 0)
        self.assertIsNone(res)

if __name__ == "__main__":
    unittest.main(verbosity=2)
