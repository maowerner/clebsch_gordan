"""Unit test for the clebsch-gordan coefficients
"""

import unittest
import numpy as np

import utils
import group_class_quat as gc
import group_cg_quat as gcg

g = [gc.TOh(irreps=True), gc.TOh(pref=np.asarray([0., 0., 1.]), irreps=True)]

@unittest.skip("skip CMF")
class TestCG_CMF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        #self.g = [gc.TOh(irreps=True)]
        self.gc = gcg.TOhCG(0, 0, 0, groups=g)
        self.p0 = np.asarray([0., 0., 0.])

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gc.p, 0)
        self.assertEqual(self.gc.p1, 0)
        self.assertEqual(self.gc.p2, 0)
        # check reference momentum
        self.assertEqual(self.gc.pref, self.p0)
        self.assertEqual(self.gc.pref1, self.p0)
        self.assertEqual(self.gc.pref2, self.p0)

    def test_momenta(self):
        # check individual momenta
        self.assertEqual(len(self.gc.momenta), 1)
        self.assertEqual(self.gc.momenta[0], self.p0)
        self.assertEqual(len(self.gc.momenta1), 1)
        self.assertEqual(self.gc.momenta1[0], self.p0)
        self.assertEqual(len(self.gc.momenta2), 1)
        self.assertEqual(self.gc.momenta2[0], self.p0)
        # check combined momenta
        self.assertEqual(len(self.gc.allmomenta), 1)
        self.assertEqual(self.gc.allmomenta[0][0], self.p0)
        self.assertEqual(self.gc.allmomenta[0][1], self.p0)
        self.assertEqual(self.gc.allmomenta[0][2], self.p0)

    def test_cosets(self):
        #res_theo = np.arange(48, dtype=int).reshape(1,48)
        # check first coset
        res1 = self.gc.gen_coset(g, 0, 0)
        # check second coset
        res2 = self.gc.gen_coset(g, 0, 0)
        self.assertEqual(res1, res2)
        self.assertEqual(res1.shape, (1,96))
        self.assertEqual(res2.shape, (1,96))

    def test_induced_representations(self):
        res_theo = np.ones((96, 1, 1), dtype=complex)
        # check the first coset
        res = self.gc.gen_ind_reps(g, 0, 0, "A1g", self.gc.coset1)
        self.assertEqual(res, res_theo)
        # check the second coset
        res = self.gc.gen_ind_reps(g, 0, 0, "A1g", self.gc.coset2)
        self.assertEqual(res, res_theo)

    def test_sort_momenta(self):
        tmp = np.zeros((3,))
        self.assertEqual(len(self.gc.smomenta1), 1)
        self.assertEqual(self.gc.smomenta1[0][0], self.p0)
        self.assertEqual(self.gc.smomenta1[0][1], 0)
        self.assertEqual(len(self.gc.smomenta2), 1)
        self.assertEqual(self.gc.smomenta2[0][0], self.p0)
        self.assertEqual(self.gc.smomenta2[0][1], 0)

    def test_check_all_cosets(self):
        res = self.gc.check_all_cosets(self.p0, self.p0, self.p0)
        self.assertEqual(res, (0,0))

    #def test_multiplicities(self):
    #    multi = self.gc.multiplicities()
    #    res_theo = np.zeros((self.gc.g.nclasses,), dtype=int)
    #    print(self.gc.g.irrepsname)
    #    print(self.gc.g.lclasses)
    #    print(self.gc.g.lelements)
    #    for irname in ["A1g", "A2g", "Ep1g"]:
    #        res_theo[self.gc.g.irrepsname.index(irname)] += 1
    #    self.assertEqual(multi, res_theo)

    def test_cg_new(self):
        self.gc.calc_cg_ha(g, self.gc.p)
        cgnames = [("A1g", 1, 1)]
        #print(self.gc.cgnames)
        #print(self.gc.cgind)
        #print(self.gc.cg)
        self.assertEqual(self.gc.cgnames, cgnames)

    def test_get_cg(self):
        res = self.gc.get_cg(self.p0, self.p0, "A1g")
        res_theo = np.ones((1,), dtype=complex)
        self.assertEqual(res, res_theo)

@unittest.skip("skip CMF reread test")
class TestCG_CMF_read(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        #self.g = [gc.TOh(irreps=True)]
        self.p0 = np.zeros((3,))
        tmp = gcg.TOhCG(0, 0, 0, groups=g)
        tmp.save()
        self.gc = gcg.TOhCG.read()

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gc.p, 0)
        self.assertEqual(self.gc.p1, 0)
        self.assertEqual(self.gc.p2, 0)
        # check reference momentum
        self.assertEqual(self.gc.pref, self.p0)
        self.assertEqual(self.gc.pref1, self.p0)
        self.assertEqual(self.gc.pref2, self.p0)

    def test_momenta(self):
        # check individual momenta
        self.assertEqual(len(self.gc.momenta), 1)
        self.assertEqual(self.gc.momenta[0], self.p0)
        self.assertEqual(len(self.gc.momenta1), 1)
        self.assertEqual(self.gc.momenta1[0], self.p0)
        self.assertEqual(len(self.gc.momenta2), 1)
        self.assertEqual(self.gc.momenta2[0], self.p0)
        # check combined momenta
        self.assertEqual(len(self.gc.allmomenta), 1)
        self.assertEqual(self.gc.allmomenta[0][0], self.p0)
        self.assertEqual(self.gc.allmomenta[0][1], self.p0)
        self.assertEqual(self.gc.allmomenta[0][2], self.p0)

    def test_cosets(self):
        #res_theo = np.arange(48, dtype=int).reshape(1,48)
        # check first coset
        res1 = self.gc.gen_coset(g, 0, 0)
        # check second coset
        res2 = self.gc.gen_coset(g, 0, 0)
        self.assertEqual(res1, res2)
        self.assertEqual(res1.shape, (1,96))
        self.assertEqual(res2.shape, (1,96))

    def test_induced_representations(self):
        res_theo = np.ones((96, 1, 1), dtype=complex)
        # check the first coset
        res = self.gc.gen_ind_reps(g, 0, 0, "A1g", self.gc.coset1)
        self.assertEqual(res, res_theo)
        # check the second coset
        res = self.gc.gen_ind_reps(g, 0, 0, "A1g", self.gc.coset2)
        self.assertEqual(res, res_theo)

    def test_sort_momenta(self):
        tmp = np.zeros((3,))
        self.assertEqual(len(self.gc.smomenta1), 1)
        self.assertEqual(self.gc.smomenta1[0][0], self.p0)
        self.assertEqual(self.gc.smomenta1[0][1], 0)
        self.assertEqual(len(self.gc.smomenta2), 1)
        self.assertEqual(self.gc.smomenta2[0][0], self.p0)
        self.assertEqual(self.gc.smomenta2[0][1], 0)

    def test_check_all_cosets(self):
        tmp = np.zeros((3,))
        res = self.gc.check_all_cosets(self.p0, self.p0, self.p0)
        self.assertEqual(res, (0,0))

    #def test_multiplicities(self):
    #    multi = self.gc.multiplicities()
    #    res_theo = np.zeros((self.gc.g.nclasses,), dtype=int)
    #    print(self.gc.g.irrepsname)
    #    print(self.gc.g.lclasses)
    #    print(self.gc.g.lelements)
    #    for irname in ["A1g", "A2g", "Ep1g"]:
    #        res_theo[self.gc.g.irrepsname.index(irname)] += 1
    #    self.assertEqual(multi, res_theo)

    def test_cg_new(self):
        self.gc.calc_cg_ha(g, self.gc.p)
        cgnames = [("A1g", 1, 1)]
        #print(self.gc.cgnames)
        #print(self.gc.cgind)
        #print(self.gc.cg)
        self.assertEqual(self.gc.cgnames, cgnames)

    def test_get_cg(self):
        res = self.gc.get_cg(self.p0, self.p0, "A1g")
        res_theo = np.ones((1,), dtype=complex)
        self.assertEqual(res, res_theo)

#@unittest.skip("skip CMF, non zero momenta")
class TestCG_CMF_non_zero_mom(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.p0 = np.asarray([0., 0., 0.])
        self.p1 = np.asarray([0., 0., 1.])
        #self.g = [gc.TOh(irreps=True), gc.TOh(pref=self.p1, irreps=True)]
        self.gc = gcg.TOhCG(0, 1, 1, groups=g)
        #self.gc = gcg.TOhCG(0, 1, 1, groups=None)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gc.p, 0)
        self.assertEqual(self.gc.p1, 1)
        self.assertEqual(self.gc.p2, 1)
        # check reference momentum
        self.assertEqual(self.gc.pref, self.p0)
        self.assertEqual(self.gc.pref1, self.p1)
        self.assertEqual(self.gc.pref2, self.p1)

    def test_momenta(self):
        tmp = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp = [np.asarray(x) for x in tmp]
        # check individual momenta
        self.assertEqual(len(self.gc.momenta), 1)
        self.assertEqual(self.gc.momenta[0], self.p0)
        self.assertEqual(len(self.gc.momenta1), 6)
        self.assertEqual(len(self.gc.momenta2), 6)
        for i, m in enumerate(tmp):
            self.assertEqual(self.gc.momenta1[i], m)
            self.assertEqual(self.gc.momenta2[i], m)
        # check combined momenta
        self.assertEqual(len(self.gc.allmomenta), 6)
        for i, m in enumerate(tmp):
            self.assertEqual(self.gc.allmomenta[i][0], self.p0)
            self.assertEqual(self.gc.allmomenta[i][1], m)
            self.assertEqual(self.gc.allmomenta[i][2], -m)

    def test_cosets(self):
        # check first coset
        res1 = self.gc.gen_coset(g, 0, 1)
        # check second coset
        res2 = self.gc.gen_coset(g, 0, 1)
        self.assertEqual(res1, res2)
        self.assertEqual(res1.shape, (6,16))
        self.assertEqual(res2.shape, (6,16))

    def test_induced_representations(self):
        res_theo = np.ones((g[0].order,), dtype=complex)*6.
        # check the first coset
        res = self.gc.gen_ind_reps(g, 0, 1, "A1g", self.gc.coset1)
        self.assertEqual(np.sum(res,axis=(1,2)), res_theo)
        # check the second coset
        res = self.gc.gen_ind_reps(g, 0, 1, "A1g", self.gc.coset2)
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

    def test_check_all_cosets(self):
        tmp0 = np.zeros((3,))
        tmp1 = np.asarray([1., 0. ,0.])
        res = self.gc.check_all_cosets(tmp0, tmp1, tmp1)
        self.assertEqual(res, (2,2))

    #def test_multiplicities(self):
    #    multi = self.gc.multiplicities()
    #    res_theo = np.zeros((self.gc.g.nclasses,), dtype=int)
    #    print(self.gc.g.irrepsname)
    #    print(self.gc.g.lclasses)
    #    print(self.gc.g.lelements)
    #    for irname in ["A1g", "A2g", "Ep1g"]:
    #        res_theo[self.gc.g.irrepsname.index(irname)] += 1
    #    self.assertEqual(multi, res_theo)

    def test_cg_new(self):
        self.gc.calc_cg_ha(g, 0)
        cgnames = [("A1g", 1, 1), ("T1u", 1, 3), ("Ep1g", 1, 2)]
        #print(self.gc.cgnames)
        #print(self.gc.cgind)
        #print(self.gc.cg)
        self.assertEqual(self.gc.cgnames, cgnames)

@unittest.skip("skip MF1")
class TestCG_MF1_one_zero(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.p0 = np.asarray([0., 0., 0.])
        self.p1 = np.asarray([0., 0., 1.])
        #self.g = [gc.TOh(irreps=True), gc.TOh(pref=self.p1, irreps=True)]
        self.gc = gcg.TOhCG(1, 1, 0, groups=g)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gc.p, 1)
        self.assertEqual(self.gc.p1, 1)
        self.assertEqual(self.gc.p2, 0)
        # check reference momentum
        self.assertEqual(self.gc.pref, self.p1)
        self.assertEqual(self.gc.pref1, self.p1)
        self.assertEqual(self.gc.pref2, self.p0)

    def test_momenta(self):
        tmp = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp = [np.asarray(x) for x in tmp]
        # check individual momenta
        self.assertEqual(len(self.gc.momenta), 6)
        self.assertEqual(len(self.gc.momenta1), 6)
        self.assertEqual(len(self.gc.momenta2), 1)
        self.assertEqual(self.gc.momenta2[0], self.p0)
        for i, m in enumerate(tmp):
            self.assertEqual(self.gc.momenta[i], m)
            self.assertEqual(self.gc.momenta1[i], m)
        # check combined momenta
        self.assertEqual(len(self.gc.allmomenta), 6)
        for i, m in enumerate(tmp):
            self.assertEqual(self.gc.allmomenta[i][0], m)
            self.assertEqual(self.gc.allmomenta[i][1], m)
            self.assertEqual(self.gc.allmomenta[i][2], self.p0)

    def test_cosets(self):
        #res_theo = np.arange(48, dtype=int).reshape(1,48)
        # check first coset
        res1 = self.gc.gen_coset(g, 0, 1)
        # check second coset
        res2 = self.gc.gen_coset(g, 0, 0)
        self.assertEqual(res1.shape, (6,16))
        self.assertEqual(res2.shape, (1,96))

    def test_induced_representations(self):
        # check the second coset
        res_theo = np.ones((96, 1, 1), dtype=complex)
        res = self.gc.gen_ind_reps(g, 0, 0, "A1g", self.gc.coset2)
        self.assertEqual(res, res_theo)
        # check the first coset
        res_theo = np.ones((g[0].order,), dtype=complex)*6.
        res = self.gc.gen_ind_reps(g, 0, 1, "A1g", self.gc.coset1)
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

    def test_check_all_cosets(self):
        tmp0 = np.zeros((3,))
        tmp1 = np.asarray([1., 0. ,0.])
        res = self.gc.check_all_cosets(tmp1, tmp1, tmp0)
        self.assertEqual(res, (2,0))

    #def test_multiplicities(self):
    #    multi = self.gc.multiplicities()
    #    res_theo = np.zeros((self.gc.g.nclasses,), dtype=int)
    #    print(self.gc.g.irrepsname)
    #    print(self.gc.g.lclasses)
    #    print(self.gc.g.lelements)
    #    for irname in ["A1g", "A2g", "Ep1g"]:
    #        res_theo[self.gc.g.irrepsname.index(irname)] += 1
    #    self.assertEqual(multi, res_theo)

    def test_cg_new(self):
        self.gc.calc_cg_ha(g, 1)
        cgnames = [("A1u", 1, 1), ("A2g", 3, 1), ("Ep1g", 1, 2)]
        #print(self.gc.cgnames)
        #print(self.gc.cgind)
        #print(self.gc.cg)
        self.assertEqual(self.gc.cgnames, cgnames)

    def test_get_cg_A1u(self):
        res = self.gc.get_cg(self.p1, self.p0, "A1u")
        res_theo = np.zeros((1,), dtype=complex)
        self.assertEqual(res, res_theo)

    def test_get_cg_A2g(self):
        res = self.gc.get_cg(self.p1, self.p0, "A2g")
        res_theo = np.zeros((3,1), dtype=complex)
        res_theo[2,0] = 1.
        self.assertEqual(res, res_theo)

if __name__ == "__main__":
    unittest.main(verbosity=2)
