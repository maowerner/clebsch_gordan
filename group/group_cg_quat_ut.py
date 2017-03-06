"""Unit test for the clebsch-gordan coefficients
"""

import unittest
import numpy as np

import utils
import group_class_quat as gc
import group_cg_quat as gcg

g = []
gt = []
def init():
    p = [None, np.asarray([0., 0., 1.])]
    S = 1./np.sqrt(2.)
    U1 = np.identity(3)
    U2 = np.asarray([[S,0,S],[0,1,0],[S,0,-S]])
    for _p in p:
        p2 = 0 if _p is None else np.dot(_p,_p)
        try:
            _g = gc.TOh.read(p2=p2)
            if np.allclose(_g.U3, U1):
                g.append(_g)
                gt.append(gc.TOh(pref=_p, irreps=True, U3=U2))
            elif np.allclose(_g.U3, U2):
                gt.append(_g)
                a = gc.TOh(pref=_p, irreps=True, U3=U1)
                a.save()
                g.append(a)
            else:
                raise IOError
        except IOError:
            a = gc.TOh(pref=_p, irreps=True, U3=U1)
            a.save()
            g.append(a)
            gt.append(gc.TOh(pref=_p, irreps=True, U3=U2))
    if not g or not gt:
        print("at least one list is empty")

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

@unittest.skip("skip CMF")
class TestCG_CMF_transformed(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        #self.g = [gc.TOh(irreps=True)]
        self.gc = gcg.TOhCG(0, 0, 0, groups=g)
        self.gct = gcg.TOhCG(0, 0, 0, groups=gt)
        self.p0 = np.asarray([0., 0., 0.])

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gct.p, 0)
        self.assertEqual(self.gct.p1, 0)
        self.assertEqual(self.gct.p2, 0)
        # check reference momentum
        self.assertEqual(self.gct.pref, self.p0)
        self.assertEqual(self.gct.pref1, self.p0)
        self.assertEqual(self.gct.pref2, self.p0)
        # transformation matrix
        S = 1./np.sqrt(2.)
        U = np.asarray([[S,0,S],[0,1,0],[S,0,-S]])
        self.assertEqual(self.gct.U0, U)
        self.assertFalse(np.allclose(self.gc.U0, U))

    def test_momenta(self):
        # check individual momenta
        self.assertEqual(len(self.gct.momenta), 1)
        self.assertEqual(self.gct.momenta[0], self.p0)
        self.assertEqual(len(self.gct.momenta1), 1)
        self.assertEqual(self.gct.momenta1[0], self.p0)
        self.assertEqual(len(self.gct.momenta2), 1)
        self.assertEqual(self.gct.momenta2[0], self.p0)
        # check combined momenta
        self.assertEqual(len(self.gct.allmomenta), 1)
        self.assertEqual(self.gct.allmomenta[0][0], self.p0)
        self.assertEqual(self.gct.allmomenta[0][1], self.p0)
        self.assertEqual(self.gct.allmomenta[0][2], self.p0)

    def test_cosets(self):
        #res_theo = np.arange(48, dtype=int).reshape(1,48)
        # check first coset
        res1 = self.gct.gen_coset(g, 0, 0)
        # check second coset
        res2 = self.gct.gen_coset(g, 0, 0)
        self.assertEqual(res1, res2)
        self.assertEqual(res1.shape, (1,96))
        self.assertEqual(res2.shape, (1,96))

    def test_induced_representations(self):
        res_theo = np.ones((96, 1, 1), dtype=complex)
        # check the first coset
        res = self.gct.gen_ind_reps(g, 0, 0, "A1g", self.gc.coset1)
        self.assertEqual(res, res_theo)
        # check the second coset
        res = self.gct.gen_ind_reps(g, 0, 0, "A1g", self.gc.coset2)
        self.assertEqual(res, res_theo)

    def test_sort_momenta(self):
        tmp = np.zeros((3,))
        self.assertEqual(len(self.gct.smomenta1), 1)
        self.assertEqual(self.gct.smomenta1[0][0], self.p0)
        self.assertEqual(self.gct.smomenta1[0][1], 0)
        self.assertEqual(len(self.gct.smomenta2), 1)
        self.assertEqual(self.gct.smomenta2[0][0], self.p0)
        self.assertEqual(self.gct.smomenta2[0][1], 0)

    def test_check_all_cosets(self):
        res = self.gct.check_all_cosets(self.p0, self.p0, self.p0)
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
        self.gct.calc_cg_ha(gt, self.gct.p)
        cgnames = [("A1g", 1, 1)]
        #print(self.gc.cgnames)
        #print(self.gc.cgind)
        #print(self.gc.cg)
        self.assertEqual(self.gct.cgnames, cgnames)

    def test_get_cg(self):
        res = self.gct.get_cg(self.p0, self.p0, "A1g")
        res_theo = np.ones((1,), dtype=complex)
        self.assertEqual(res, res_theo)

    def test_cg_transformed(self):
        res1 = self.gc.get_cg(self.p0, self.p0, "T1u")
        res2 = self.gct.get_cg(self.p0, self.p0, "T1u")
        self.assertEqual(res1, res2)

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

@unittest.skip("skip CMF, non zero momenta")
class TestCG_CMF_non_zero_mom(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.p0 = np.asarray([0., 0., 0.])
        self.p1 = np.asarray([0., 0., 1.])
        self.gc = gcg.TOhCG(0, 1, 1, groups=g)
        self.U = np.asarray([[0.,-1.j,0.],[0.,0.,1.],[-1.,0.,0.]])

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gc.p, 0)
        self.assertEqual(self.gc.p1, 1)
        self.assertEqual(self.gc.p2, 1)
        # check reference momentum
        self.assertEqual(self.gc.pref, self.U.dot(self.p0))
        self.assertEqual(self.gc.pref1, self.U.dot(self.p1))
        self.assertEqual(self.gc.pref2, self.U.dot(self.p1))

    def test_momenta(self):
        tmp = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp = [self.U.dot(np.asarray(x)) for x in tmp]
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

        tmp1 = [3, 4, 1, 0, 5, 2]
        tmp2 = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp2 = [self.U.dot(np.asarray(x)) for x in tmp2]
        for i in range(len(self.gc.smomenta1)):
            self.assertEqual(self.gc.smomenta1[i][0], tmp2[i])
            self.assertEqual(self.gc.smomenta1[i][1], tmp1[i])
            self.assertEqual(self.gc.smomenta2[i][0], tmp2[i])
            self.assertEqual(self.gc.smomenta2[i][1], tmp1[i])

    def test_check_all_cosets(self):
        tmp0 = np.zeros((3,))
        tmp1 = self.U.dot(np.asarray([1., 0. ,0.]))
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
        self.assertEqual(self.gc.cgnames, cgnames)


@unittest.skip("skip CMF, non zero momenta")
class TestCG_CMF_non_zero_mom_transformed(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.p0 = np.asarray([0., 0., 0.])
        self.p1 = np.asarray([0., 0., 1.])
        self.gc = gcg.TOhCG(0, 1, 1, groups=gt)
        s = 1./np.sqrt(2.)
        self.U0 = np.asarray([[0.,-1.j,0.],[0.,0.,1.],[-1.,0.,0.]])
        self.U = np.asarray([[-s,-1.j*s,0.],[0.,0.,1.],[s,-1.j*s,0.]])

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        # check p2
        self.assertEqual(self.gc.p, 0)
        self.assertEqual(self.gc.p1, 1)
        self.assertEqual(self.gc.p2, 1)
        # check reference momentum
        self.assertEqual(self.gc.pref, self.U.dot(self.p0))
        self.assertEqual(self.gc.pref1, self.U.dot(self.p1))
        self.assertEqual(self.gc.pref2, self.U.dot(self.p1))

    def test_momenta(self):
        tmp = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp = [self.U.dot(np.asarray(x)) for x in tmp]
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

        tmp1 = [3, 4, 1, 0, 5, 2]
        tmp2 = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp2 = [self.U.dot(np.asarray(x)) for x in tmp2]
        for i in range(len(self.gc.smomenta1)):
            self.assertEqual(self.gc.smomenta1[i][0], tmp2[i])
            self.assertEqual(self.gc.smomenta1[i][1], tmp1[i])
            self.assertEqual(self.gc.smomenta2[i][0], tmp2[i])
            self.assertEqual(self.gc.smomenta2[i][1], tmp1[i])

    def test_check_all_cosets(self):
        tmp0 = np.zeros((3,))
        tmp1 = self.U.dot(np.asarray([1., 0. ,0.]))
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
        self.assertEqual(self.gc.cgnames, cgnames)

@unittest.skip("skip MF1")
class TestCG_MF1_one_zero(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.U = np.asarray([[0.,-1.j,0.],[0.,0.,1.],[-1.,0.,0.]])
        self.p0 = np.asarray([0., 0., 0.])
        self.p1 = self.U.dot(np.asarray([0., 0., 1.]))
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
        tmp = [self.U.dot(np.asarray(x)) for x in tmp]
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

        tmp1 = [3, 4, 1, 0, 5, 2]
        tmp2 = [[-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]]
        tmp2 = [self.U.dot(np.asarray(x)) for x in tmp2]
        for i in range(len(self.gc.smomenta1)):
            self.assertEqual(self.gc.smomenta1[i][0], tmp2[i])
            self.assertEqual(self.gc.smomenta1[i][1], tmp1[i])

    def test_check_all_cosets(self):
        tmp0 = np.zeros((3,))
        tmp1 = self.U.dot(np.asarray([1., 0. ,0.]))
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

    def test_get_cg_A1g(self):
        res = self.gc.get_cg(self.p1, self.p0, "A1g")
        self.assertIsNone(res)

    def test_get_cg_A1u(self):
        res = self.gc.get_cg(self.p1, self.p0, "A1u")
        res_theo = np.ones((1,), dtype=complex)*-0.5
        self.assertEqual(res, res_theo)

    def test_get_cg_A2g(self):
        res = self.gc.get_cg(self.p1, self.p0, "A2g")
        res_theo = np.zeros((3,1), dtype=complex)
        res_theo[0,0] = 0.5
        self.assertEqual(res, res_theo)

    def test_get_cg_A2u(self):
        res = self.gc.get_cg(self.p1, self.p0, "A2u")
        self.assertIsNone(res)

class TestCG_CMF_Print(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gc = gcg.TOhCG(1, 1, 0, groups=g)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_print(self):
        self.gc.print_operators()

if __name__ == "__main__":
    init()
    unittest.main(verbosity=2)
