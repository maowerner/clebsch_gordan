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
        np.set_printoptions(suppress=True)

    def test_1D(self):
        elements = [quat.QNew.create_from_vector(quat.qPar[x], 1) for x in range(5)]
        res1 = gg.genJ0(elements)
        res2 = gg.gen1D(elements)
        res_theo = np.ones_like(res1)
        self.assertEqual(res1, res_theo)
        self.assertEqual(res2, res_theo)
        self.assertEqual(res1, res2)

    def test_1D_inv(self):
        inv = [1., -1., 1., -1., 1]
        elements = [quat.QNew.create_from_vector(quat.qPar[x], inv[x]) for x in range(5)]
        res1 = gg.genJ0(elements, inv=True)
        res2 = gg.gen1D(elements, inv=True)
        res_theo = np.asarray(inv).reshape(-1,1,1)
        self.assertEqual(res1, res_theo)
        self.assertEqual(res2, res_theo)
        self.assertEqual(res1, res2)

    def test_2D(self):
        elements = [quat.QNew.create_from_vector(quat.qPar[x], 1) for x in range(5)]
        res1 = gg.genJ1_2(elements)
        res2 = gg.gen2D(elements)
        #res_theo = np.ones_like(res1)
        #self.assertEqual(res1, res_theo)
        #self.assertEqual(res2, res_theo)
        self.assertEqual(res1, res2)

    def test_2D_inv(self):
        inv = [1., -1., 1., -1., 1]
        elements = [quat.QNew.create_from_vector(quat.qPar[x], inv[x]) for x in range(5)]
        res1 = gg.genJ1_2(elements)
        res2 = gg.gen2D(elements)
        #res_theo = np.ones_like(res1)
        #self.assertEqual(res1, res_theo)
        #self.assertEqual(res2, res_theo)
        self.assertEqual(res1, res2)

    def test_3D(self):
        elements = [quat.QNew.create_from_vector(quat.qPar[x], 1) for x in range(5)]
        res1 = gg.genJ1(elements)
        res2 = gg.gen3D(elements)
        res_theo = np.ones_like(res1)
        #self.assertEqual(res1, res_theo)
        #self.assertEqual(res2, res_theo)
        for i in range(res1.shape[0]):
            tmpmsg = "element %d failed:\n%r\n\n%r" % (i, res1[i], res2[i])
            self.assertEqual(res1[i], res2[i], msg=tmpmsg)
        # gen3D fails

    def test_4D(self):
        elements = [quat.QNew.create_from_vector(quat.qPar[x], 1) for x in range(5)]
        res1 = gg.genJ3_2(elements)
        res2 = gg.gen4D(elements)
        res_theo = np.ones_like(res1)
        #self.assertEqual(res1, res_theo)
        #self.assertEqual(res2, res_theo)
        for i in range(res1.shape[0]):
            tmpmsg = "element %d failed:\n%r\n\n%r" % (i, res1[i], res2[i])
            self.assertEqual(res1[i], res2[i], msg=tmpmsg)
        #self.assertEqual(res1, res2)
        # gen4D fails

if __name__ == "__main__":
    unittest.main(verbosity=2)

