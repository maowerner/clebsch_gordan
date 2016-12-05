"""Unit test for the group class
"""

import unittest
import numpy as np

import utils
import quat
import group_generators_quat as gg

class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_J0(self):
        elements = [quat.QNew.create_from_vector(quat.qPar[x], 1) for x in range(5)]
        res = gg.genJ0(elements)
        res_theo = np.ones_like(res)
        self.assertEqual(res, res_theo)

    def test_J1_2(self):
        elements = [quat.QNew.create_from_vector(quat.qPar[x], 1) for x in range(5)]
        res = gg.genJ1_2(elements)
        res_theo = np.ones_like(res)
        self.assertEqual(res, res_theo)

    def test_J1(self):
        elements = [quat.QNew.create_from_vector(quat.qPar[x], 1) for x in range(5)]
        res = gg.genJ1(elements)
        res_theo = np.ones_like(res)
        self.assertEqual(res, res_theo)

    def test_J3_2(self):
        elements = [quat.QNew.create_from_vector(quat.qPar[x], 1) for x in range(5)]
        res = gg.genJ3_2(elements)
        res_theo = np.ones_like(res)
        self.assertEqual(res, res_theo)

if __name__ == "__main__":
    unittest.main(verbosity=2)

