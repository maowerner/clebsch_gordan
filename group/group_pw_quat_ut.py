"""Unit test for the partial-wave based operators
"""

import unittest
import numpy as np

import utils
import group_pw_quat as pw
import group_class_quat as gc

try:
    g0 = gc.TOh.read(p2=0)
except IOError:
    g0 = gc.TOh(irreps=True)
    g0.save()
try:
    g1 = gc.TOh.read(p2=1)
except IOError:
    g1 = gc.TOh(irreps=True)
    g1.save()

class TestPWOpsCMF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pwop = pw.PWOps(groups=[g0, g1], p=0, p1=1, p2=1, s2=0)
        self.p0 = np.asarray([0., 0., 0.])
        self.p1 = np.asarray([0., 0., 1.])

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_init(self):
        pwop = pw.PWOps()
        self.assertIsNotNone(pwop)

    def test_momenta(self):
        tmp = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp = [np.asarray(x) for x in tmp]
        # check individual momenta
        # check combined momenta
        self.assertEqual(len(self.pwop.allmomenta), 6)
        for i, m in enumerate(tmp):
            self.assertEqual(self.pwop.allmomenta[i][0], self.p0)
            self.assertEqual(self.pwop.allmomenta[i][1], m)
            self.assertEqual(self.pwop.allmomenta[i][2], -m)

    def test_calc_component(self):
        res = self.pwop.calc_component(0,0,0,0,0,0)
        #print(res)

    def test_calc_index(self):
        ind = self.pwop.rot_index
        res_theo = np.ones((self.pwop.elength,)) * np.sum(range(self.pwop.mlength))
        res = np.sum(ind, axis=0)
        self.assertEqual(res, res_theo)

    def test_calc_op(self):
        res = self.pwop.calc_op(0,0,0,0)
        #print(res.shape)
        res_theo = np.ones((6,))
        res_theo /= np.sqrt(res_theo.size)
        self.assertEqual(res, res_theo)

    #def test_print_op(self):
    #    self.pwop.print_op(1,0,1,0)

    def test_print_all_jmax2(self):
        self.pwop.print_all(3)

    def test_get_all_ops_j1(self):
        res, ind = self.pwop.get_all_ops(1)
        res_theo = np.ones((1,6))
        res_theo /= np.sqrt(res_theo.size)
        self.assertEqual(res, res_theo)

#@unittest.skip("skip CMF scalar-vector")
class TestPWOpsCMF_SV(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pwop = pw.PWOps(groups=[g0, g1], p=0, p1=1, p2=1, s2=1)
        self.p0 = np.asarray([0., 0., 0.])
        self.p1 = np.asarray([0., 0., 1.])

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_init(self):
        pwop = pw.PWOps()
        self.assertIsNotNone(pwop)

    def test_momenta(self):
        tmp = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp = [np.asarray(x) for x in tmp]
        # check individual momenta
        # check combined momenta
        self.assertEqual(len(self.pwop.allmomenta), 6)
        for i, m in enumerate(tmp):
            self.assertEqual(self.pwop.allmomenta[i][0], self.p0)
            self.assertEqual(self.pwop.allmomenta[i][1], m)
            self.assertEqual(self.pwop.allmomenta[i][2], -m)

    def test_calc_component(self):
        res = self.pwop.calc_component(0,0,0,0,0,0)
        #print(res)

    def test_calc_index(self):
        ind = self.pwop.rot_index
        res_theo = np.ones((self.pwop.elength,)) * np.sum(range(self.pwop.mlength))
        res = np.sum(ind, axis=0)
        self.assertEqual(res, res_theo)

    def test_calc_op(self):
        res = self.pwop.calc_op(0,0,0,0)
        #print(res.shape)
        res_theo = np.zeros((6,1,3))
        self.assertEqual(res, res_theo)

    #def test_print_op(self):
    #    self.pwop.print_op(1,0,1,0)

    def test_print_all_jmax2(self):
        self.pwop.print_all(2)

#@unittest.skip("skip MF1")
class TestPWOpsMF1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pwop = pw.PWOps(groups=[g0, g1])
        self.p0 = np.asarray([0., 0., 0.])
        self.p1 = np.asarray([0., 0., 1.])

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_init(self):
        pwop = pw.PWOps()
        self.assertIsNotNone(pwop)

    def test_momenta(self):
        tmp = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp = [np.asarray(x) for x in tmp]
        # check individual momenta
        # check combined momenta
        self.assertEqual(len(self.pwop.allmomenta), 6)
        for i, m in enumerate(tmp):
            self.assertEqual(self.pwop.allmomenta[i][0], m)
            self.assertEqual(self.pwop.allmomenta[i][1], m)
            self.assertEqual(self.pwop.allmomenta[i][2], self.p0)

    def test_calc_component(self):
        res = self.pwop.calc_component(0,0,0,0,0,0)
        res_theo = np.ones((6,))*16/np.sqrt(4*np.pi)
        self.assertEqual(res, res_theo)

    def test_calc_index(self):
        ind = self.pwop.rot_index
        res_theo = np.ones((self.pwop.elength,)) * np.sum(range(self.pwop.mlength))
        res = np.sum(ind, axis=0)
        self.assertEqual(res, res_theo)

    def test_calc_op(self):
        res = self.pwop.calc_op(0,0,0,0)
        #print(res.shape)
        res_theo = np.ones((6,))
        res_theo /= np.sqrt(res_theo.size)
        self.assertEqual(res, res_theo)

    #def test_print_op(self):
    #    self.pwop.print_op(1,0,1,0)

    def test_print_all(self):
        self.pwop.print_all(1)

    def test_print_all_jmax2(self):
        self.pwop.print_all(2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
