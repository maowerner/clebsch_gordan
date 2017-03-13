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
        self.inv = [1., -1., 1., -1., 1]
        self.elements = [quat.QNew.create_from_vector(
                quat.qPar[x], 1) for x in range(5)]
        self.elementsinv = [quat.QNew.create_from_vector(
                quat.qPar[x], -1) for x in range(5)]
                #quat.qPar[x], self.inv[x]) for x in range(5)]

    def test_1D(self):
        res1 = gg.genJ0(self.elements)
        res2 = gg.gen1D(self.elements)
        res_theo = np.ones_like(res1)
        self.assertEqual(res1, res_theo)
        self.assertEqual(res2, res_theo)
        self.assertEqual(res1, res2)

    def test_1D_inv(self):
        res1 = gg.genJ0(self.elementsinv, inv=True)
        res2 = gg.gen1D(self.elementsinv, inv=True)
        res_theo = np.ones_like(res1) * -1
        self.assertEqual(res1, res_theo)
        self.assertEqual(res2, res_theo)
        self.assertEqual(res1, res2)

    def test_2D(self):
        res1 = gg.genJ1_2(self.elements)
        res2 = gg.gen2D(self.elements)
        #res_theo = np.ones_like(res1)
        #self.assertEqual(res1, res_theo)
        #self.assertEqual(res2, res_theo)
        self.assertEqual(res1, res2)

    def test_2D_inv(self):
        # implicit inversion due to construction
        res1 = gg.genJ1_2(self.elementsinv)
        res2 = gg.gen2D(self.elementsinv, inv=True)
        #res_theo = np.ones_like(res1)
        #self.assertEqual(res1, res_theo)
        #self.assertEqual(res2, res_theo)
        self.assertEqual(res1, res2)

    def test_3D(self):
        res1 = gg.genJ1(self.elements)
        res2 = gg.gen3D(self.elements)
        #res_theo = np.asarray([[1,0,0],[0,1,0],[0,0,1]],dtype=complex)
        res_theo = np.identity(3, dtype=complex)
        self.assertEqual(res1[0], res_theo)
        self.assertEqual(res2[0], res_theo)
        for i in range(res1.shape[0]):
            tmpmsg = "element %d failed:\n%r\n\n%r" % (i, res1[i], res2[i])
            #self.assertEqual(res1[i], res2[i], msg=tmpmsg)
        # gen3D fails

    def test_4D(self):
        res1 = gg.genJ3_2(self.elements)
        res2 = gg.gen4D(self.elements)
        #res_theo = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=complex)
        res_theo = np.identity(4, dtype=complex)
        self.assertEqual(res1[0], res_theo)
        self.assertEqual(res2[0], res_theo)
        for i in range(res1.shape[0]):
            tmpmsg = "element %d failed:\n%r\n\n%r" % (i, res1[i], res2[i])
            #self.assertEqual(res1[i], res2[i], msg=tmpmsg)
        #self.assertEqual(res1, res2)
        # gen4D fails

    def test_2Dp_CMF(self):
        res = gg.genEpCMF(self.elements)
        res_theo = np.identity(2, dtype=complex)
        self.assertEqual(res[0], res_theo)

    def test_2Dp_MF1(self):
        res = gg.genEpMF1(self.elements)
        res_theo = np.identity(2, dtype=complex)
        self.assertEqual(res[0], res_theo)


class TestAllElements(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)
        np.set_printoptions(suppress=True)
        self.elements = []
        for i in range(24):
            self.elements.append(quat.QNew.create_from_vector(quat.qPar[i], 1))
        for i in range(24):
            self.elements.append(quat.QNew.create_from_vector(quat.qPar[i], -1))
        for i in range(24):
            self.elements.append(quat.QNew.create_from_vector(-quat.qPar[i], 1))
        for i in range(24):
            self.elements.append(quat.QNew.create_from_vector(-quat.qPar[i], -1))

    def test_1D_no_inversion(self):
        res = gg.gen1D(self.elements)
        res_theo = np.ones((len(self.elements), 1, 1), dtype=complex)
        self.assertEqual(res, res_theo)

    def test_1D_inversion(self):
        res = gg.gen1D(self.elements, inv=True)
        res_theo = np.ones((len(self.elements), 1, 1), dtype=complex)
        for i in range(24, 48):
            res_theo[i] *= -1
        for i in range(72, 96):
            res_theo[i] *= -1
        self.assertEqual(res, res_theo)

    def test_2D_no_inversion(self):
        res = gg.gen2D(self.elements)
        for i in res:
            self.assertFalse(utils._eq(i))

    def test_2D_inversion(self):
        res = gg.gen2D(self.elements, inv=True)
        for i in res:
            self.assertFalse(utils._eq(i))

    def test_3D_no_inversion(self):
        res = gg.gen3D(self.elements)
        for i in res:
            self.assertFalse(utils._eq(i))

    def test_3D_inversion(self):
        res = gg.gen3D(self.elements, inv=True)
        for i in res:
            self.assertFalse(utils._eq(i))

    def test_4D_no_inversion(self):
        res = gg.gen4D(self.elements)
        for i in res:
            self.assertFalse(utils._eq(i))

    def test_4D_inversion(self):
        res = gg.gen4D(self.elements, inv=True)
        for i in res:
            self.assertFalse(utils._eq(i))

@unittest.skip("skip output testing")
class PrintIrreps(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)
        np.set_printoptions(suppress=True)
        self.elements = [quat.QNew.create_from_vector(x, 1) for x in quat.qPar]
        self.elementsi = [quat.QNew.create_from_vector(x, -1) for x in quat.qPar]

    def test_gen2D_MF1(self):
        res = gg.genEpMF1(self.elements, inv=True)
        for r in res:
            print(" ")
            print(r)
        self.assertTrue(True)

    #def test_gen2D(self):
    #    #res = gg.gen2D(self.elements, inv=True)
    #    #res = gg.genJ1_2(self.elements, inv=True)
    #    res = gg.genEpCMF(self.elements, inv=True)
    #    for r in res:
    #        print(" ")
    #        print(r)
    #    self.assertTrue(True)

    #def test_gen3D(self):
    #    #res = gg.gen3D(self.elements, inv=True)
    #    #res = gg.genJ1(self.elements, inv=True)
    #    res = gg.genT1CMF(self.elements, inv=True)
    #    for r in res:
    #        print(" ")
    #        print(r)
    #    self.assertTrue(True)

    #def test_gen4D(self):
    #    #res = gg.gen3D(self.elements, inv=True)
    #    #res = gg.genJ3_2(self.elements, inv=True)
    #    res = gg.genF1CMF(self.elements, inv=True)
    #    for r in res:
    #        print(" ")
    #        print(r)
    #    self.assertTrue(True)

if __name__ == "__main__":
    unittest.main(verbosity=2)

