"""Unit test for the group class
"""

import unittest
import numpy as np

import utils
import group_class_quat as gc
import group_basis_quat as gb

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

#@unittest.skip("skip base tests")
class TestBasis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.basis = gb.TOhBasis(g0)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_init(self):
        self.assertIsNotNone(self.basis)

    def test_wrong_init(self):
        self.assertRaises(RuntimeError, gb.TOhBasis, gc.TOh())

    def test_get_basis_vecs_A1g_j0(self):
        ir = g0.irreps[0]
        bvec = self.basis.get_basis_vecs(ir, 0, 0, 0)
        res_theo = np.ones((1,1))
        self.assertEqual(bvec, res_theo)

    def test_get_basis_vec_A1g_j4(self):
        ir = g0.irreps[0]
        bvec = self.basis.get_basis_vecs(ir, 0, 0, 4)
        res_theo = np.zeros((9,))
        res_theo[0] = (5./24.)
        res_theo[4] = np.sqrt(35./288.)
        res_theo[8] = (5./24.)
        self.assertEqual(bvec[0], res_theo)

    def test_multiplicityO3(self):
        irnames = g0.irrepsname
        res_theo = np.zeros((len(irnames),))
        # J=0, assume only A1g contained
        res_theo[irnames.index("A1g")] = 1.
        res = self.basis.multiplicityO3(g0, 0)
        self.assertEqual(res, res_theo)
        # J=1, assume only T1u
        res_theo.fill(0.)
        res_theo[irnames.index("T1u")] = 1.
        res = self.basis.multiplicityO3(g0, 1)
        self.assertEqual(res, res_theo)

    def test_calculate(self):
        ind1 = g0.irrepsname.index("A1g")
        ind2 = g0.irrepsname.index("T1u")
        res1, res2 = self.basis.calculate(g0, 1)
        s = 1./np.sqrt(2.)
        m0 = np.ones((1,))
        m1 = np.asarray([[s,0,s],[0,1,0],[s,0,-s]])
        dim = g0.irrepdim

        self.assertEqual(len(res1), 9)
        self.assertEqual(len(res2), 9)
        for i, (r1, r2) in enumerate(zip(res1[:2], res2[:2])):
            for j, (_r1, _r2) in enumerate(zip(r1, r2)):
                if i == 0 and j == ind1:
                    self.assertEqual(_r1, m0)
                    self.assertEqual(_r2, 1)
                elif i == 1 and j == ind2:
                    self.assertEqual(_r1, m1)
                    self.assertEqual(_r2, 1)
                else:
                    self.assertIsNone(_r1)
                    self.assertEqual(_r2, 0)
        for i, (r1, r2) in enumerate(zip(res1, res2)):
            mult = np.asarray(self.basis.multiplicityO3(g0, i))
            try:
                self.assertEqual(mult, np.asarray(r2))
                for j, m in enumerate(mult):
                    if m == 0:
                        self.assertIsNone(r1[j])
                    else:
                        self.assertEqual(r1[j].shape[0], m*dim[j]) 
            except:
                print("j=%d" % i)
                raise

    def test_calc_basis_vec_A1g_j0(self):
        ir = g0.irreps[g0.irrepsname.index("A1g")]
        res = self.basis.calc_basis_vec(ir, 0, 1)
        res_theo = np.ones((1,))
        self.assertEqual(res, res_theo)
        self.assertEqual(res.shape[0], 1)

    def test_calc_basis_vec_T1u_j1(self):
        ir = g0.irreps[g0.irrepsname.index("T1u")]
        res = self.basis.calc_basis_vec(ir, 1, 3)
        s = 1./np.sqrt(2.)
        res_theo = np.asarray([[s,0,s],[0,1,0],[s,0,-s]])
        self.assertEqual(res, res_theo)
        self.assertEqual(res.shape[0], 3)

    def test_calc_basis_vec_T1u_j5(self):
        ir = g0.irreps[g0.irrepsname.index("T1u")]
        res = self.basis.calc_basis_vec(ir, 5, 3)
        res_theo = np.zeros((11,))
        res_theo[0] = 0.51538820
        res_theo[2] = -0.20337230
        res_theo[4] = 0.43933439
        res_theo[6] = -res_theo[4]
        res_theo[8] = -res_theo[2]
        res_theo[10] = -res_theo[0]
        self.assertEqual(res[2], res_theo)
        self.assertEqual(res.shape[0], 6)

    def test_calc_basis_vec_T1g_j4(self):
        ir = g0.irreps[g0.irrepsname.index("A1g")]
        res = self.basis.calc_basis_vec(ir, 4, 1)
        res_theo = np.zeros((9,))
        res_theo[[0,-1]] = np.sqrt(5./24.)
        res_theo[4] = np.sqrt(7./12.)
        self.assertEqual(res[0], res_theo)
        self.assertEqual(res.shape[0], 1)

    def test_max_j(self):
        res = self.basis.get_max_j(g0, 1)
        self.assertEqual(res, 9)
        res = self.basis.get_max_j(g0, 2)
        self.assertEqual(res, 13)

#@unittest.skip("skip base tests, LG(1)")
class TestBasis_LG1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.basis = gb.TOhBasis(g1)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_init(self):
        self.assertIsNotNone(self.basis)

    def test_max_j(self):
        res = self.basis.get_max_j(g1, 1)
        self.assertEqual(res, 4)
        res = self.basis.get_max_j(g1, 2)
        self.assertEqual(res, 5)

    def test_calc_basis_vec_A1g_j0(self):
        ir = g1.irreps[g1.irrepsname.index("A1g")]
        res = self.basis.calc_basis_vec(ir, 0, 1)
        res_theo = np.ones((1,))
        self.assertEqual(res, res_theo)
        self.assertEqual(res.shape[0], 1)

    def test_calc_basis_vec_A1u_j1(self):
        ir = g1.irreps[g1.irrepsname.index("A1g")]
        res = self.basis.calc_basis_vec(ir, 1, 1)
        res_theo = np.zeros((3,))
        res_theo[1] = 1.
        self.assertEqual(res, res_theo)
        self.assertEqual(res.shape[0], 1)

    def test_calc_basis_vec_A2u_j2(self):
        ir = g1.irreps[g1.irrepsname.index("A2u")]
        res = self.basis.calc_basis_vec(ir, 2, 1)
        res_theo = np.zeros((5,))
        res_theo[[0,-1]] = 1./np.sqrt(2.)
        self.assertEqual(res, res_theo)
        self.assertEqual(res.shape[0], 1)

    def test_calc_basis_vec_A2g_j4(self):
        ir = g1.irreps[g1.irrepsname.index("A2u")]
        self.assertRaises(RuntimeError, self.basis.calc_basis_vec, ir, 4, 1)
        #res = self.basis.calc_basis_vec(ir, 4, 1)
        #res_theo = np.zeros((9,))
        #res_theo[[2,-3]] = 1./np.sqrt(2.)
        #self.assertEqual(res, res_theo)
        #self.assertEqual(res.shape[0], 1)

    def test_calc_basis_vec_Ep1g_j1(self):
        ir = g1.irreps[g1.irrepsname.index("Ep1g")]
        res = self.basis.calc_basis_vec(ir, 1, 2)
        res_theo = np.zeros((2,3))
        res_theo[0,0] = 1.
        res_theo[1,2] = -1.
        self.assertEqual(res, res_theo)
        self.assertEqual(res.shape[0], 2)

#@unittest.skip("bla")
class TestBasisPrint(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.basis = gb.TOhBasis(g0)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_print_table(self):
        self.basis.print_table()

    def test_print_overview(self):
        self.basis.print_overview()

#@unittest.skip("bla")
class TestBasisLG1Print(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.basis = gb.TOhBasis(g1)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_print_table(self):
        self.basis.print_table()

    def test_print_overview(self):
        self.basis.print_overview()

if __name__ == "__main__":
    unittest.main(verbosity=2)
