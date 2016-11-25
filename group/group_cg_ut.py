"""Unit test for the clebsch-gordan coefficients
"""

import unittest
import numpy as np

import utils
import group_class as gc
import group_cg as gcg

@unittest.skip("testing the other class")
class TestCG_CMF(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)
        self.g = [gc.OhGroup(instances=True)]
        self.gc = gcg.OhCG(0, 0, 0, groups=self.g)

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
        # check indices
        self.assertEqual(self.gc.i1, -1)
        self.assertEqual(self.gc.i2, -1)
        self.assertEqual(self.gc.mu1, 0)
        self.assertEqual(self.gc.mu2, 0)
        self.assertEqual(len(self.gc.i1i2), 1)
        self.assertEqual(self.gc.i1i2[0][0], tmp)
        self.assertEqual(self.gc.i1i2[0][1:], (-1, -1))

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
        res_theo = np.arange(48, dtype=int).reshape(1,48)
        # check first coset
        res = self.gc.gen_coset(self.gc.g1)
        self.assertEqual(res, res_theo)
        # check second coset
        res = self.gc.gen_coset(self.gc.g2)
        self.assertEqual(res, res_theo)

    def test_induced_representations(self):
        res_theo = np.ones((48, 1, 1), dtype=complex)
        # check the first coset
        res = self.gc.gen_ind_reps(self.gc.g, self.gc.g1, "A1",
                self.gc.coset1)
        self.assertEqual(res, res_theo)
        # check the second coset
        res = self.gc.gen_ind_reps(self.gc.g, self.gc.g2, "A1",
                self.gc.coset2)
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
        self.assertEqual(len(res), 48)
        self.assertTrue(np.all(res))
        res = self.gc.check_coset(self.gc.pref2, self.gc.pref2,
                self.gc.coset2[0])
        self.assertEqual(len(res), 48)
        self.assertTrue(np.all(res))

    def test_check_all_cosets(self):
        tmp = np.zeros((3,))
        res = self.gc.check_all_cosets(tmp, tmp, tmp)
        self.assertEqual(res, (0,0,-1,-1))

    def test_calc_pion_cg_A1(self):
        tmp = np.zeros((3,))
        res = self.gc.calc_pion_cg(tmp, tmp, tmp, "A1")
        cg_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_A2(self):
        tmp = np.zeros((3,))
        res = self.gc.calc_pion_cg(tmp, tmp, tmp, "A1")
        cg_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_E(self):
        tmp = np.zeros((3,))
        res = self.gc.calc_pion_cg(tmp, tmp, tmp, "E")
        cg_theo = np.zeros((2,2), dtype=complex)
        self.assertEqual(res, cg_theo, msg="might fail due to phase")
        #self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_T1(self):
        tmp = np.zeros((3,))
        res = self.gc.calc_pion_cg(tmp, tmp, tmp, "T1")
        cg_theo = np.zeros((3,3), dtype=complex)
        self.assertEqual(res, cg_theo, msg="might fail due to phase")
        #self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_T2(self):
        tmp = np.zeros((3,))
        res = self.gc.calc_pion_cg(tmp, tmp, tmp, "T2")
        cg_theo = np.zeros((3,3), dtype=complex)
        self.assertEqual(res, cg_theo, msg="might fail due to phase")
        #self.assertEqual(res, cg_theo)

    def test_get_pion_cg_A1(self):
        res = self.gc.get_pion_cg("A1")
        res_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "A1")
        self.assertEqual(res[1], res_theo)
        # res[2] is already checked in test_momenta

    def test_get_pion_cg_A1_twice(self):
        res1 = self.gc.get_pion_cg("A1")
        res2 = self.gc.get_pion_cg("A1")
        self.assertIs(res1[1], res2[1])

    def test_norm_cgs_1(self):
        # since the norm is coupled to the total momentum
        # no further tests can be made here
        data = np.ones((1,1,1), dtype=complex)
        res = self.gc._norm_cgs(data)
        self.assertEqual(res, data[:-1])

class TestCG_CMF_non_zero_mom(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = [gc.OhGroup(instances=True), gc.OhGroup(p2=1, instances=True)]
        self.gc = gcg.OhCG(0, 1, 1, groups=self.g)

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
        # check indices
        self.assertEqual(self.gc.i1, -1)
        self.assertEqual(self.gc.i2, -1)
        self.assertEqual(self.gc.mu1, 0)
        self.assertEqual(self.gc.mu2, 0)
        self.assertEqual(len(self.gc.i1i2), 1)
        tmp = np.asarray([0., 0., 0.])
        self.assertEqual(self.gc.i1i2[0][0], tmp)
        self.assertEqual(self.gc.i1i2[0][1:], (-1, -1))

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
        res_theo = np.asarray(
                [[0,3,6,9,12,15,18,1,2,4,5,37,38,43,44,47],
                 [7,35,41,19,24,28,31,13,42,10,36,33,30,21,26,16],
                 [8,40,46,20,25,29,32,39,14,45,11,27,22,23,34,17]], dtype=int)

        # check first coset
        res = self.gc.gen_coset(self.gc.g1)
        self.assertEqual(res, res_theo)
        # check second coset
        res = self.gc.gen_coset(self.gc.g2)
        self.assertEqual(res, res_theo)

    def test_induced_representations(self):
        res_theo = np.ones((48,), dtype=complex)*3.
        # check the first coset
        res = self.gc.gen_ind_reps(self.gc.g, self.gc.g1, "A1",
                self.gc.coset1)
        self.assertEqual(np.sum(res,axis=(1,2)), res_theo)
        # check the second coset
        res = self.gc.gen_ind_reps(self.gc.g, self.gc.g2, "A1",
                self.gc.coset2)
        self.assertEqual(np.sum(res,axis=(1,2)), res_theo)

    def test_sort_momenta(self):
        tmp = np.zeros((3,))
        self.assertEqual(len(self.gc.smomenta1), 6)
        self.assertEqual(self.gc.smomenta1[0][0], tmp)
        self.assertEqual(self.gc.smomenta1[0][1], 0)
        self.assertEqual(len(self.gc.smomenta2), 6)
        self.assertEqual(self.gc.smomenta2[0][0], tmp)
        self.assertEqual(self.gc.smomenta2[0][1], 0)

    def test_check_coset_1(self):
        # check first subset of elements
        res = self.gc.check_coset(self.gc.pref1, self.gc.pref1,
                self.gc.coset1[0])
        self.assertEqual(len(res), 16)
        self.assertTrue(np.all(res))
        # check second subset of elements
        res = self.gc.check_coset(self.gc.pref1, self.gc.pref1,
                self.gc.coset1[1])
        self.assertFalse(np.all(res))
        # check second subset of elements
        res = self.gc.check_coset(self.gc.pref1, self.gc.pref1,
                self.gc.coset1[2])
        self.assertFalse(np.all(res))
        # check if the negative vector is in the same coset
        res = self.gc.check_coset(-self.gc.pref1, self.gc.pref1,
                self.gc.coset1[0])
        self.assertTrue(np.all(res))
        # check second subset of elements
        res = self.gc.check_coset(-self.gc.pref1, self.gc.pref1,
                self.gc.coset1[1])
        self.assertFalse(np.all(res))
        # check second subset of elements
        res = self.gc.check_coset(-self.gc.pref1, self.gc.pref1,
                self.gc.coset1[2])
        self.assertFalse(np.all(res))

    def test_check_coset_2(self):
        # check first subset of elements
        res = self.gc.check_coset(self.gc.pref2, self.gc.pref2,
                self.gc.coset2[0])
        self.assertEqual(len(res), 16)
        self.assertTrue(np.all(res))
        # check second subset of elements
        res = self.gc.check_coset(self.gc.pref2, self.gc.pref2,
                self.gc.coset2[1])
        self.assertEqual(len(res), 16)
        self.assertFalse(np.all(res))
        # check second subset of elements
        res = self.gc.check_coset(self.gc.pref2, self.gc.pref2,
                self.gc.coset2[2])
        self.assertEqual(len(res), 16)
        self.assertFalse(np.all(res))
        # check if the negative vector is in the same coset
        res = self.gc.check_coset(-self.gc.pref2, self.gc.pref2,
                self.gc.coset2[0])
        self.assertEqual(len(res), 16)
        self.assertTrue(np.all(res))
        # check second subset of elements
        res = self.gc.check_coset(-self.gc.pref2, self.gc.pref2,
                self.gc.coset2[1])
        self.assertEqual(len(res), 16)
        self.assertFalse(np.all(res))
        # check second subset of elements
        res = self.gc.check_coset(-self.gc.pref2, self.gc.pref2,
                self.gc.coset2[2])
        self.assertEqual(len(res), 16)
        self.assertFalse(np.all(res))

    def test_check_all_cosets(self):
        tmp = np.zeros((3,))
        res = self.gc.check_all_cosets(tmp, tmp, tmp)
        self.assertEqual(res, (0,0,-1,-1))

    def test_calc_pion_cg_A1(self):
        tmp1 = np.zeros((3,))
        tmp2 = np.asarray([0.,0.,1.])
        res = self.gc.calc_pion_cg(tmp1, tmp2, -tmp2, "A1")
        cg_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_A2(self):
        tmp1 = np.zeros((3,))
        tmp2 = np.asarray([0.,0.,1.])
        res = self.gc.calc_pion_cg(tmp1, tmp2, -tmp2, "A2")
        cg_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_E(self):
        tmp1 = np.zeros((3,))
        tmp2 = np.asarray([0.,0.,1.])
        res = self.gc.calc_pion_cg(tmp1, tmp2, -tmp2, "E")
        cg_theo = np.zeros((2,2), dtype=complex)
        self.assertEqual(res, cg_theo, msg="might fail due to phase")
        #self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_T1(self):
        tmp1 = np.zeros((3,))
        tmp2 = np.asarray([0.,0.,1.])
        res = self.gc.calc_pion_cg(tmp1, tmp2, -tmp2, "T1")
        cg_theo = np.zeros((3,3), dtype=complex)
        self.assertEqual(res, cg_theo, msg="might fail due to phase")
        #self.assertEqual(res, cg_theo)

    def test_calc_pion_cg_T2(self):
        tmp1 = np.zeros((3,))
        tmp2 = np.asarray([0.,0.,1.])
        res = self.gc.calc_pion_cg(tmp1, tmp2, -tmp2, "T2")
        cg_theo = np.zeros((3,3), dtype=complex)
        self.assertEqual(res, cg_theo, msg="might fail due to phase")
        #self.assertEqual(res, cg_theo)

    def test_get_pion_cg_A1(self):
        res = self.gc.get_pion_cg("A1")
        res_theo = np.ones((1,1), dtype=complex)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0], "A1")
        self.assertEqual(res[1], res_theo)
        # res[2] is already checked in test_momenta

    def test_get_pion_cg_A1_twice(self):
        res1 = self.gc.get_pion_cg("A1")
        res2 = self.gc.get_pion_cg("A1")
        self.assertIs(res1[1], res2[1])

    def test_norm_cgs_1(self):
        # since the norm is coupled to the total momentum
        # no further tests can be made here
        data = np.ones((6,1,1), dtype=complex)/np.sqrt(6)
        res = self.gc._norm_cgs(data)
        self.assertEqual(res, data[:-1])

if __name__ == "__main__":
    unittest.main(verbosity=2)
