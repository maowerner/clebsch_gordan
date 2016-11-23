"""Unit test for the group generators
"""

import unittest
import numpy as np

import group_generators as gg
from rotations import _all_rotations as ar

_prec = 1e-10
class TestGenG1(unittest.TestCase):
    def setUp(self):
        self.c = [2., 0., np.sqrt(2), -np.sqrt(2),
                1., -1., 0., -2.]
        self.gen = lambda x: gg.gen_G1(x)

    def test_class_I(self):
        mx = self.gen(ar[0])
        self.assertAlmostEqual(np.trace(mx), self.c[0], delta=_prec)

    def test_class_6C4(self):
        mx = self.gen(ar[1])
        self.assertAlmostEqual(np.trace(mx), self.c[1], delta=_prec)

    def test_class_6C4p(self):
        mx = self.gen(ar[7])
        self.assertAlmostEqual(np.trace(mx), self.c[2], delta=_prec)

    def test_class_6C8(self):
        mx = self.gen(ar[13])
        self.assertAlmostEqual(np.trace(mx), self.c[3], delta=_prec)

    def test_class_8C6(self):
        mx = self.gen(ar[19])
        self.assertAlmostEqual(np.trace(mx), self.c[4], delta=_prec)

    def test_class_8C3(self):
        mx = self.gen(ar[27])
        self.assertAlmostEqual(np.trace(mx), self.c[5], delta=_prec)

    def test_class_12C4p(self):
        mx = self.gen(ar[35])
        self.assertAlmostEqual(np.trace(mx), self.c[6], delta=_prec)

    def test_class_J(self):
        mx = self.gen(ar[47])
        self.assertAlmostEqual(np.trace(mx), self.c[7], delta=_prec)

class TestGenT1(unittest.TestCase):
    def setUp(self):
        self.c = [3., -1., 1., 1., 0., 0., -1., 3.]
        self.gen = lambda x: gg.gen_T1(x)

    def test_class_I(self):
        mx = self.gen(ar[0])
        self.assertAlmostEqual(np.trace(mx), self.c[0], delta=_prec)

    def test_class_6C4(self):
        mx = self.gen(ar[1])
        self.assertAlmostEqual(np.trace(mx), self.c[1], delta=_prec)

    def test_class_6C4p(self):
        mx = self.gen(ar[7])
        self.assertAlmostEqual(np.trace(mx), self.c[2], delta=_prec)

    def test_class_6C8(self):
        mx = self.gen(ar[13])
        self.assertAlmostEqual(np.trace(mx), self.c[3], delta=_prec)

    def test_class_8C6(self):
        mx = self.gen(ar[19])
        self.assertAlmostEqual(np.trace(mx), self.c[4], delta=_prec)

    def test_class_8C3(self):
        mx = self.gen(ar[27])
        self.assertAlmostEqual(np.trace(mx), self.c[5], delta=_prec)

    def test_class_12C4p(self):
        mx = self.gen(ar[35])
        self.assertAlmostEqual(np.trace(mx), self.c[6], delta=_prec)

    def test_class_J(self):
        mx = self.gen(ar[47])
        self.assertAlmostEqual(np.trace(mx), self.c[7], delta=_prec)

if __name__ == "__main__":
    unittest.main(verbosity=2)
