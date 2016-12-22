"""Unit test for the clebsch-gordan coefficients
"""

import unittest
import numpy as np

import utils
import group_class_quat as gc
import group_cg_quat as gcg

@unittest.skip("skip CMF")
class TestCG_CMF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = [gc.TOh(irreps=True)]
        self.gc = gcg.TOhCG(0, 0, 0, groups=self.g)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gc.p, 0)
        self.assertEqual(self.gc.p1, 0)
        self.assertEqual(self.gc.p2, 0)
        # check reference momentum
        tmp = np.asarray([0., 0., 0.])
        self.assertEqual(self.gc.pref, tmp)
        self.assertEqual(self.gc.pref1, tmp)
        self.assertEqual(self.gc.pref2, tmp)
        # check groups selected
        self.assertEqual(self.gc.g, self.g[0])
        self.assertEqual(self.gc.g0, self.g[0])
        self.assertEqual(self.gc.g1, self.g[0])
        self.assertEqual(self.gc.g2, self.g[0])
        # check result arrays
        self.assertEqual(self.gc.irreps, [])
        self.assertEqual(self.gc.cgs, [])

    def test_momenta(self):
        tmp = np.asarray([0., 0., 0.])
        # check individual momenta
        self.assertEqual(len(self.gc.momenta), 1)
        self.assertEqual(self.gc.momenta[0], tmp)
        self.assertEqual(len(self.gc.momenta1), 1)
        self.assertEqual(self.gc.momenta1[0], tmp)
        self.assertEqual(len(self.gc.momenta2), 1)
        self.assertEqual(self.gc.momenta2[0], tmp)
        # check combined momenta
        self.assertEqual(len(self.gc.allmomenta), 1)
        self.assertEqual(self.gc.allmomenta[0][0], tmp)
        self.assertEqual(self.gc.allmomenta[0][1], tmp)
        self.assertEqual(self.gc.allmomenta[0][2], tmp)

    def test_cosets(self):
        #res_theo = np.arange(48, dtype=int).reshape(1,48)
        # check first coset
        res1 = self.gc.gen_coset(self.gc.g1)
        # check second coset
        res2 = self.gc.gen_coset(self.gc.g2)
        self.assertEqual(res1, res2)
        self.assertEqual(res1.shape, (1,96))
        self.assertEqual(res2.shape, (1,96))

    def test_induced_representations(self):
        res_theo = np.ones((96, 1, 1), dtype=complex)
        # check the first coset
        res = self.gc.gen_ind_reps(self.gc.g1, "A1g", self.gc.coset1)
        self.assertEqual(res, res_theo)
        # check the second coset
        res = self.gc.gen_ind_reps(self.gc.g2, "A1g", self.gc.coset2)
        self.assertEqual(res, res_theo)

    def test_sort_momenta(self):
        tmp = np.zeros((3,))
        self.assertEqual(len(self.gc.smomenta1), 1)
        self.assertEqual(self.gc.smomenta1[0][0], tmp)
        self.assertEqual(self.gc.smomenta1[0][1], 0)
        self.assertEqual(len(self.gc.smomenta2), 1)
        self.assertEqual(self.gc.smomenta2[0][0], tmp)
        self.assertEqual(self.gc.smomenta2[0][1], 0)

    def test_check_coset(self):
        res = self.gc.check_coset(self.gc.pref1, self.gc.pref1,
                self.gc.coset1[0])
        self.assertEqual(len(res), 96)
        self.assertTrue(np.all(res))
        res = self.gc.check_coset(self.gc.pref2, self.gc.pref2,
                self.gc.coset2[0])
        self.assertEqual(len(res), 96)
        self.assertTrue(np.all(res))

    def test_check_all_cosets(self):
        tmp = np.zeros((3,))
        res = self.gc.check_all_cosets(tmp, tmp, tmp)
        self.assertEqual(res, (0,0))

    def test_calc_pion_cg_zero(self):
        tmp = np.zeros((3,))
        irnames = ["A1u", "A2g", "A2u", "E1g", "E1u", "E2g", "E2u",
                "T1g", "T1u", "T2g", "T2u", "Ep1g", "Ep1u", "G1g", "G1u"]
        for irn in irnames:
            dim = 0
            if irn[0] in ["A", "K"]:
                dim = 1
            elif irn[0] == "E":
                dim = 2
            elif irn[0] == "T":
                dim = 3
            elif irn[0] == "G":
                dim = 4
            else:
                raise RuntimeError("cannot handle irrep %s" % irn)
            res = self.gc.calc_pion_cg(tmp, tmp, tmp, irn)
            cg_theo = np.zeros((dim, dim), dtype=complex)
            msg = "irrep %s:\nresult\n%r\n\nexpected:\n%r" % (irn, res, cg_theo)
            self.assertEqual(res, cg_theo, msg=msg)

    def test_calc_pion_cg_A1g(self):
        tmp = np.zeros((3,))
        res = self.gc.calc_pion_cg(tmp, tmp, tmp, "A1g")
        cg_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(res, cg_theo)

    def test_get_pion_cg_A1g(self):
        res = self.gc.get_pion_cg("A1g")
        res_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "A1g")
        self.assertEqual(res[1], res_theo)
        # res[2] is already checked in test_momenta

    def test_get_pion_cg_A1g_twice(self):
        res1 = self.gc.get_pion_cg("A1g")
        res2 = self.gc.get_pion_cg("A1g")
        self.assertIs(res1[1], res2[1])

    def test_get_pion_cg_zero(self):
        irnames = ["A1u", "A2g", "A2u", "E1g", "E1u", "E2g", "E2u",
                "T1g", "T1u", "T2g", "T2u", "Ep1g", "Ep1u", "G1g", "G1u"]
        for irn in irnames:
            res = self.gc.get_pion_cg(irn)
            msg = "irrep %s failed" % irn
            self.assertEqual(res[0], irn)
            self.assertIsNone(res[1], msg=msg)

    def test_norm_cgs_1(self):
        # since the norm is coupled to the total momentum
        # no further tests can be made here
        data = np.ones((1,1,1), dtype=complex)
        res = self.gc._norm_cgs(data)
        self.assertEqual(res, data[:-1])

@unittest.skip("skip CMF, non zero momenta")
class TestCG_CMF_non_zero_mom(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.p = np.asarray([0., 0., 1.])
        self.g = [gc.TOh(irreps=True), gc.TOh(pref=self.p, irreps=True)]
        self.gc = gcg.TOhCG(0, 1, 1, groups=self.g)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gc.p, 0)
        self.assertEqual(self.gc.p1, 1)
        self.assertEqual(self.gc.p2, 1)
        # check reference momentum
        tmp = np.asarray([0., 0., 0.])
        self.assertEqual(self.gc.pref, tmp)
        tmp = np.asarray([0., 0., 1.])
        self.assertEqual(self.gc.pref1, tmp)
        self.assertEqual(self.gc.pref2, tmp)
        # check groups selected
        self.assertEqual(self.gc.g, self.g[0])
        self.assertEqual(self.gc.g0, self.g[0])
        self.assertEqual(self.gc.g1, self.g[1])
        self.assertEqual(self.gc.g2, self.g[1])
        # check result arrays
        self.assertEqual(self.gc.irreps, [])
        self.assertEqual(self.gc.cgs, [])

    def test_momenta(self):
        tmp1 = np.asarray([0., 0., 0.])
        tmp2 = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp2 = [np.asarray(x) for x in tmp2]
        # check individual momenta
        self.assertEqual(len(self.gc.momenta), 1)
        self.assertEqual(self.gc.momenta[0], tmp1)
        self.assertEqual(len(self.gc.momenta1), 6)
        self.assertEqual(len(self.gc.momenta2), 6)
        for i, m in enumerate(tmp2):
            self.assertEqual(self.gc.momenta1[i], m)
            self.assertEqual(self.gc.momenta2[i], m)
        # check combined momenta
        self.assertEqual(len(self.gc.allmomenta), 6)
        for i, m in enumerate(tmp2):
            self.assertEqual(self.gc.allmomenta[i][0], tmp1)
            self.assertEqual(self.gc.allmomenta[i][1], m)
            self.assertEqual(self.gc.allmomenta[i][2], -m)

    def test_cosets(self):
        # check first coset
        res1 = self.gc.gen_coset(self.gc.g1)
        # check second coset
        res2 = self.gc.gen_coset(self.gc.g2)
        self.assertEqual(res1, res2)
        self.assertEqual(res1.shape, (6,16))
        self.assertEqual(res2.shape, (6,16))

    def test_induced_representations(self):
        res_theo = np.ones((self.g[0].order,), dtype=complex)*6.
        # check the first coset
        res = self.gc.gen_ind_reps(self.gc.g1, "A1g", self.gc.coset1)
        self.assertEqual(np.sum(res,axis=(1,2)), res_theo)
        # check the second coset
        res = self.gc.gen_ind_reps(self.gc.g2, "A1g", self.gc.coset2)
        self.assertEqual(np.sum(res,axis=(1,2)), res_theo)

    def test_sort_momenta(self):
        self.assertEqual(len(self.gc.smomenta1), 6)
        self.assertEqual(len(self.gc.smomenta2), 6)

        tmp1 = [3, 5, 1, 0, 4, 2]
        tmp2 = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp2 = [np.asarray(x) for x in tmp2]
        for i in range(len(self.gc.smomenta1)):
            self.assertEqual(self.gc.smomenta1[i][0], tmp2[i])
            self.assertEqual(self.gc.smomenta1[i][1], tmp1[i])
            self.assertEqual(self.gc.smomenta2[i][0], tmp2[i])
            self.assertEqual(self.gc.smomenta2[i][1], tmp1[i])

    def test_check_coset_1(self):
        # check first subset of elements
        res = self.gc.check_coset(self.gc.pref1, self.gc.pref1,
                self.gc.coset1[0])
        self.assertEqual(len(res), 16)
        self.assertTrue(np.all(res))
        # check all other subject
        for i in range(1, len(self.gc.coset1)):
            res = self.gc.check_coset(self.gc.pref1, self.gc.pref1,
                    self.gc.coset1[i])
            self.assertFalse(np.all(res))

    def test_check_coset_2(self):
        # check first subset of elements
        res = self.gc.check_coset(self.gc.pref2, self.gc.pref2,
                self.gc.coset2[0])
        self.assertEqual(len(res), 16)
        self.assertTrue(np.all(res))
        # check all other subject
        for i in range(1, len(self.gc.coset2)):
            res = self.gc.check_coset(self.gc.pref2, self.gc.pref2,
                    self.gc.coset2[i])
            self.assertFalse(np.all(res))

    def test_check_all_cosets(self):
        tmp0 = np.zeros((3,))
        tmp1 = np.asarray([1., 0. ,0.])
        res = self.gc.check_all_cosets(tmp0, tmp1, tmp1)
        self.assertEqual(res, (2,2))

    def test_calc_pion_cg_A1g(self):
        tmp1 = np.zeros((3,))
        tmp2 = np.asarray([0.,0.,1.])
        res = self.gc.calc_pion_cg(tmp1, tmp2, -tmp2, "A1g")
        cg_theo = np.ones((1,1), dtype=complex)/6.
        self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_Ep1g(self):
        tmp1 = np.zeros((3,))
        tmp2 = np.asarray([0.,0.,1.])
        res = self.gc.calc_pion_cg(tmp1, tmp2, -tmp2, "Ep1g")
        cg_theo = np.ones((2,2),dtype=complex)
        cg_theo[:,1] = -0.5+0.8660254j
        self.assertEqual(res/res[0,0], cg_theo)

    def test_calc_pion_cg_T1u(self):
        tmp1 = np.zeros((3,))
        tmp2 = np.asarray([0.,0.,1.])
        res = self.gc.calc_pion_cg(tmp1, tmp2, -tmp2, "T1u")
        cg_theo = np.zeros((3,3), dtype=complex)
        cg_theo[1,0] = 1./np.sqrt(8)
        cg_theo[1,2] = -1./np.sqrt(8)
        self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_zero(self):
        tmp = np.zeros((3,))
        tmp1 = np.zeros((3,))
        tmp1[2] = 1.
        irnames = ["A1u", "A2g", "A2u", "E1g", "E1u", "E2g", "E2u",
                "T1g", "T2g", "T2u", "Ep1u", "G1g", "G1u"]
        for irn in irnames:
            dim = 0
            if irn[0] in ["A", "K"]:
                dim = 1
            elif irn[0] == "E":
                dim = 2
            elif irn[0] == "T":
                dim = 3
            elif irn[0] == "G":
                dim = 4
            else:
                raise RuntimeError("cannot handle irrep %s" % irn)
            res = self.gc.calc_pion_cg(tmp, tmp1, -tmp1, irn)
            cg_theo = np.zeros((dim, dim), dtype=complex)
            msg = "irrep %s:\nresult\n%r\n\nexpected:\n%r" % (irn, res, cg_theo)
            self.assertEqual(res, cg_theo, msg=msg)

    def test_get_pion_cg_A1g(self):
        res = self.gc.get_pion_cg("A1g")
        res_theo = np.ones((6,1), dtype=complex)/np.sqrt(6)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "A1g")
        self.assertEqual(res[1], res_theo)
        # res[2] is already checked in test_momenta

    def test_get_pion_cg_zero(self):
        irnames = ["A1u", "A2g", "A2u", "E1g", "E1u", "E2g", "E2u",
                "T1g", "T2g", "T2u", "Ep1u", "G1g", "G1u"]
        for irn in irnames:
            res = self.gc.get_pion_cg(irn)
            msg = "irrep %s failed" % irn
            self.assertEqual(res[0], irn)
            self.assertIsNone(res[1], msg=msg)

    def test_get_pion_cg_A1g_twice(self):
        res1 = self.gc.get_pion_cg("A1g")
        res2 = self.gc.get_pion_cg("A1g")
        self.assertIs(res1[1], res2[1])

    def test_norm_cgs_1(self):
        # since the norm is coupled to the total momentum
        # no further tests can be made here
        data = np.ones((6,1,1), dtype=complex)/np.sqrt(6)
        res = self.gc._norm_cgs(data)
        self.assertEqual(res, data)

#@unittest.skip("skip MF1")
class TestCG_MF1_one_zero(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.p = np.asarray([0., 0., 1.])
        self.g = [gc.TOh(irreps=True), gc.TOh(pref=self.p, irreps=True)]
        self.gc = gcg.TOhCG(1, 1, 0, groups=self.g)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gc.p, 1)
        self.assertEqual(self.gc.p1, 1)
        self.assertEqual(self.gc.p2, 0)
        # check reference momentum
        tmp = np.asarray([0., 0., 0.])
        tmp1 = np.asarray([0., 0., 1.])
        self.assertEqual(self.gc.pref, tmp1)
        self.assertEqual(self.gc.pref1, tmp1)
        self.assertEqual(self.gc.pref2, tmp)
        # check groups selected
        self.assertEqual(self.gc.g, self.g[1])
        self.assertEqual(self.gc.g0, self.g[0])
        self.assertEqual(self.gc.g1, self.g[1])
        self.assertEqual(self.gc.g2, self.g[0])
        # check result arrays
        self.assertEqual(self.gc.irreps, [])
        self.assertEqual(self.gc.cgs, [])

    def test_momenta(self):
        tmp1 = np.asarray([0., 0., 0.])
        tmp2 = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp2 = [np.asarray(x) for x in tmp2]
        # check individual momenta
        self.assertEqual(len(self.gc.momenta), 6)
        self.assertEqual(len(self.gc.momenta1), 6)
        self.assertEqual(len(self.gc.momenta2), 1)
        self.assertEqual(self.gc.momenta2[0], tmp1)
        for i, m in enumerate(tmp2):
            self.assertEqual(self.gc.momenta[i], m)
            self.assertEqual(self.gc.momenta1[i], m)
        # check combined momenta
        self.assertEqual(len(self.gc.allmomenta), 6)
        for i, m in enumerate(tmp2):
            self.assertEqual(self.gc.allmomenta[i][0], m)
            self.assertEqual(self.gc.allmomenta[i][1], m)
            self.assertEqual(self.gc.allmomenta[i][2], tmp1)

    def test_cosets(self):
        #res_theo = np.arange(48, dtype=int).reshape(1,48)
        # check first coset
        res1 = self.gc.gen_coset(self.gc.g1)
        # check second coset
        res2 = self.gc.gen_coset(self.gc.g2)
        self.assertEqual(res1.shape, (6,16))
        self.assertEqual(res2.shape, (1,96))

    def test_induced_representations(self):
        # check the second coset
        res_theo = np.ones((96, 1, 1), dtype=complex)
        res = self.gc.gen_ind_reps(self.gc.g2, "A1g", self.gc.coset2)
        self.assertEqual(res, res_theo)
        # check the first coset
        res_theo = np.ones((self.g[0].order,), dtype=complex)*6.
        res = self.gc.gen_ind_reps(self.gc.g1, "A1g", self.gc.coset1)
        self.assertEqual(np.sum(res,axis=(1,2)), res_theo)

    def test_sort_momenta(self):
        tmp = np.zeros((3,))
        self.assertEqual(len(self.gc.smomenta1), 6)
        self.assertEqual(len(self.gc.smomenta2), 1)
        self.assertEqual(self.gc.smomenta2[0][0], tmp)
        self.assertEqual(self.gc.smomenta2[0][1], 0)

        tmp1 = [3, 5, 1, 0, 4, 2]
        tmp2 = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp2 = [np.asarray(x) for x in tmp2]
        for i in range(len(self.gc.smomenta1)):
            self.assertEqual(self.gc.smomenta1[i][0], tmp2[i])
            self.assertEqual(self.gc.smomenta1[i][1], tmp1[i])

    def test_check_coset_1(self):
        # check first subset of elements
        res = self.gc.check_coset(self.gc.pref1, self.gc.pref1,
                self.gc.coset1[0])
        self.assertEqual(len(res), 16)
        self.assertTrue(np.all(res))
        # check all other subject
        for i in range(1, len(self.gc.coset1)):
            res = self.gc.check_coset(self.gc.pref1, self.gc.pref1,
                    self.gc.coset1[i])
            self.assertFalse(np.all(res))

    def test_check_coset_2(self):
        # check first subset of elements
        res = self.gc.check_coset(self.gc.pref2, self.gc.pref2,
                self.gc.coset2[0])
        self.assertEqual(len(res), 96)
        self.assertTrue(np.all(res))
        # check all other subject
        for i in range(1, len(self.gc.coset2)):
            res = self.gc.check_coset(self.gc.pref2, self.gc.pref2,
                    self.gc.coset2[i])
            self.assertFalse(np.all(res))

    def test_check_all_cosets(self):
        tmp0 = np.zeros((3,))
        tmp1 = np.asarray([1., 0. ,0.])
        res = self.gc.check_all_cosets(tmp1, tmp1, tmp0)
        self.assertEqual(res, (2,0))

    def test_calc_pion_cg_zero(self):
        tmp = np.zeros((3,))
        tmp1 = np.asarray([0., 0., 1.])
        irnames = ["A1g", "A1u", "A2u", "E1g", "E1u", "E2g", "E2u",
                "T1g", "T1u", "T2g", "T2u", "Ep1g", "Ep1u", "G1g", "G1u"]
        for irn in irnames:
            dim = 0
            if irn[0] in ["A", "K"]:
                dim = 1
            elif irn[0] == "E":
                dim = 2
            elif irn[0] == "T":
                dim = 3
            elif irn[0] == "G":
                dim = 4
            else:
                raise RuntimeError("cannot handle irrep %s" % irn)
            res = self.gc.calc_pion_cg(tmp1, tmp1, tmp, irn)
            cg_theo = np.zeros((dim, dim), dtype=complex)
            msg = "irrep %s:\nresult\n%r\n\nexpected:\n%r" % (irn, res, cg_theo)
            self.assertEqual(res, cg_theo, msg=msg)

    def test_calc_pion_cg_A2g(self):
        tmp = np.zeros((3,))
        tmp1 = np.asarray([0., 0., 1.])
        res = self.gc.calc_pion_cg(tmp1, tmp1, tmp, "A2g")
        cg_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(res, cg_theo)

    def test_get_pion_cg_A2g(self):
        res = self.gc.get_pion_cg("A2g")
        res_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "A2g")
        self.assertEqual(res[1], res_theo)
        # res[2] is already checked in test_momenta

    def test_get_pion_cg_A2g_twice(self):
        res1 = self.gc.get_pion_cg("A2g")
        res2 = self.gc.get_pion_cg("A2g")
        self.assertIs(res1[1], res2[1])

    def test_get_pion_cg_zero(self):
        irnames = ["A1g", "A1u", "A2u", "E1g", "E1u", "E2g", "E2u",
                "T1g", "T1u", "T2g", "T2u", "Ep1g", "Ep1u", "G1g", "G1u"]
        for irn in irnames:
            res = self.gc.get_pion_cg(irn)
            msg = "irrep %s failed" % irn
            self.assertEqual(res[0], irn)
            self.assertIsNone(res[1], msg=msg)

    def test_norm_cgs_1(self):
        # since the norm is coupled to the total momentum
        # no further tests can be made here
        data = np.ones((1,1,1), dtype=complex)
        res = self.gc._norm_cgs(data)
        self.assertEqual(res, data[:-1])

if __name__ == "__main__":
    unittest.main(verbosity=2)
