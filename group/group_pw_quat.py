"""Class for the partial-wave based operators."""

import numpy as np
import itertools as it
import scipy.special as sp
from sympy import N
from sympy.physics.quantum.cg import CG

import quat
import utils
import group_generators_quat as gen

class PWOps(object):
    def __init__(self, groups=None, p=1, p1=1, p2=0, s1=0, s2=0, prec=1e-6):
        self.prec = prec
        self.p = p
        self.p1 = p1
        self.p2 = p2
        self.s1 = s1
        self.s2 = s2
        self.gen_momenta()
        if groups is None:
            return
        pind = [x.p2 for x in groups]
        self.elements = groups[pind.index(p)].elements
        self.elength = len(self.elements)
        self.calc_index()

    def gen_momenta(self):
        pm = 4 # maximum component in each direction
        def _abs(x):
            return np.vdot(x, x) <= pm
        def _abs1(x,a):
            return np.vdot(x, x) == a
        gen = it.ifilter(_abs, it.product(range(-pm,pm+1), repeat=3))
        lp3 = [np.asarray(y, dtype=int) for y in gen]
        momenta =  [y for y in it.ifilter(lambda x: _abs1(x,self.p), lp3)]
        momenta1 = [y for y in it.ifilter(lambda x: _abs1(x,self.p1), lp3)]
        momenta2 = [y for y in it.ifilter(lambda x: _abs1(x,self.p2), lp3)]
        # check allowed momentum combinations
        self.allmomenta = []
        for p in momenta:
            for p1 in momenta1:
                for p2 in momenta2:
                    if utils._eq(p1+p2-p):
                        self.allmomenta.append((p, p1, p2))
        self.mlength = len(self.allmomenta)
        if not self.allmomenta:
            raise RuntimeError("no valid momentum combination found")

    def calc_index(self):
        def _r1(k):
            kn = np.vdot(k, k)
            if kn < self.prec or self.p1 == 0:
                return 0., 0.
            theta = np.arccos(k[2]/kn)
            phi = np.arctan2(k[1], k[0])
            return theta, phi
        #self.rot_index = np.zeros((1, self.elength), dtype=int)
        self.angles = np.zeros((self.mlength, self.elength, 2))
        self.rot_index = np.zeros((self.mlength, self.elength), dtype=int)
        # use the T1u irrep for the rotations, change to non-symmetric
        # spherical harmonics base
        s = 1./np.sqrt(2.)
        U = np.asarray([[0,-1.j,0],[0,0,1],[-1,0,0]])
        Ui = U.conj().T
        rot = gen.genT1CMF(self.elements, inv=True)
        lpref = [[0,0,0],[0,0,1],[1,1,0],[1,1,1]]
        pref = np.asarray(lpref[self.p1])
        for i, q in enumerate(rot):
            #_p1 = Ui.dot(q.dot(U.dot(pref))).real
            #theta, phi = _r1(_p1)
            #for l, (_k, k1, k2) in enumerate(self.allmomenta):
            #    if utils._eq(_p1, k1):
            #        self.rot_index[0,i] = l
            #        self.angles[l] = np.asarray([theta,phi])
            #        break
            for j, (_p, _p1, _p2) in enumerate(self.allmomenta):
                p = Ui.dot(q.dot(U.dot(_p)))
                p1 = Ui.dot(q.dot(U.dot(_p1))).real
                ang = _r1(p1)
                p2 = Ui.dot(q.dot(U.dot(_p2)))
                if not utils._eq(_p1+_p2-_p):
                    raise RuntimeError("some rotation does not conserve momentum!")
                for l, (_k, _k1, _k2) in enumerate(self.allmomenta):
                    if utils._eq(p1, _k1) and utils._eq(p2, _k2):
                        self.rot_index[j,i] = l
                        self.angles[j,i] = np.asarray(ang)
                        break

    def calc_component(self, j, mj, l, ml, s, ms):
        def _r1(rm, p, p1, p2):
            if self.p1 == 0:
                return 0., 0.
            k = rm.dot(p1)
            kn = np.vdot(k, k)
            theta = np.arccos(k[2]/kn)
            phi = np.arctan2(k[1], k[0])
            return theta, phi

        _s1, _s2 = int(2*self.s1+1), int(2*self.s2+1)
        # sympy uses its on version of float, convert to normal floats
        c1 = float(CG(s, ms, l, ml, j, mj).doit())
        res = np.zeros((self.mlength,_s1, _s2), dtype=complex)
        for m1, m2 in it.product(range(_s1), range(_s2)):
            c2 = float(CG(self.s1, m1-self.s1, self.s2, m2-self.s2, s, ms).doit())
            #for i, ang in zip(self.rot_index[0], self.angles):
            #    res[i,m1,m2] += c2*sp.sph_harm(ml, l, ang[0], ang[1]).conj()
            for i, q in enumerate(self.elements):
                #rm = q.rotation_matrix(inv=True)
                for j, pvecs in enumerate(self.allmomenta):
                    #theta, phi = _r1(rm, *pvecs)
                    _r = self.rot_index[j, i]
                    theta, phi = self.angles[j,i]
                    res[_r,m1,m2] += c2*sp.sph_harm(ml, l, theta, phi).conj()
        return c1*res

    def calc_op(self, j, mj, l, s):
        _s1, _s2 = int(2*self.s1+1), int(2*self.s2+1)
        c = np.zeros((self.mlength,_s1, _s2), dtype=complex)

        for ml in range(-l,l+1):
            for ms in range(-s, s+1):
                c += self.calc_component(j, mj, l, ml, s, ms)

        # normalize the vectors
        #for m1, m2 in it.product(range(_s1), range(_s2)):
        #    n = np.sqrt(np.vdot(c[:,m1,m2], c[:,m1,m2]))
        #    if n < self.prec:
        #        continue
        #    c[:,m1,m2] /= n
        return c

    def print_head(self, colsize=7):
        def momtostring(p):
            return " (%+1d,%+1d,%+1d)" % (p[0],p[1],p[2])
        s, s1, s2 = [], [], []
        for p, p1, p2 in self.allmomenta:
            s1.append(momtostring(p1))
            s2.append(momtostring(p2))
            s.append(momtostring(p))
        s1 = "|".join(s1)
        s1 = "|".join(["p1".center(colsize), s1])
        s2 = "|".join(s2)
        s2 = "|".join(["p2".center(colsize), s2])
        s = "|".join(s)
        if colsize == 23:
            s = "|".join([" J | mJ| L | S | m1| m2", s])
        else:
            s = "|".join(["P".center(colsize), s])
        print(s1)
        print(s2)
        print(s)

    def print_coeffs(self, coeffs, rowname=None):
        def cgtostring(cg):
            _n = np.absolute(cg) < self.prec
            _r = np.absolute(cg.real) < self.prec
            _i = np.absolute(cg.imag) < self.prec
            if _n:
                return "%11d" % 0
            elif _r and not _i:
                return "%+ 10.3fi" % cg.imag
            elif not _r and _i:
                return "%+ 10.3f " % cg.real
            else:
                return "%+.2f%+.2fi" % (cg.real, cg.imag)
        _s1, _s2 = int(2*self.s1+1), int(2*self.s2+1)
        for m1, m2 in it.product(range(_s1), range(_s2)):
            if rowname is None:
                tmp = ["%3d"%(m1-self.s1),"%3d"%(m2-self.s2)] 
                tmp = "|".join(tmp)
            else:
                tmp = ["%3d"%x for x in rowname]
                tmp.append("%3d"%(m1-self.s1))
                tmp.append("%3d"%(m2-self.s2))
                tmp = "|".join(tmp)
            s = [cgtostring(x) for x in coeffs[:,m1,m2]]
            s = "|".join(s)
            s = "|".join([tmp, s])
            print(s)

    def print_coeffs_new(self, coeffs, rowname=None):
        def cgtostring(cg):
            _n = np.absolute(cg) < self.prec
            _r = np.absolute(cg.real) < self.prec
            _i = np.absolute(cg.imag) < self.prec
            if _n:
                return "%11d" % 0
            elif _r and not _i:
                return "%+ 10.3fi" % cg.imag
            elif not _r and _i:
                return "%+ 10.3f " % cg.real
            else:
                return "%+.2f%+.2fi" % (cg.real, cg.imag)
        _s1, _s2 = int(2*self.s1+1), int(2*self.s2+1)
        tmp = ["%3d"%x for x in rowname]
        tmp = "|".join(tmp)
        s = [cgtostring(x) for x in coeffs]
        s = "|".join(s)
        s = "|".join([tmp, s])
        print(s)

    def print_op(self, j, mj, l, _s):
        res = self.calc_op(j, mj, l, _s)
        line = "-" * (5+len(self.allmomenta)*12)
        print("O_s=%d(p_1) x O_s=%d(p_2) -> O(P)_{S,L}^{J,m_J}"%(
                self.s1, self.s2))
        self.print_head()
        print(line)
        # print the operator
        self.print_coeffs(res)
        print(line)

    def get_all_ops(self, jmax):
        res = []
        ind = []
        srange = self.s1+self.s2+1
        _s1, _s2 = int(2*self.s1+1), int(2*self.s2+1)
        for j in range(jmax):
            jmult = int(2*j+1)
            for mj in range(jmult):
                for s in range(srange):
                    for l in range(j+srange):
                        c = self.calc_op(j, mj, l, s)
                        if utils._eq(c):
                            continue
                        # orthonormalize to previos vectors
                        for m1, m2 in it.product(range(_s1), range(_s2)):
                            _c = c[:,m1,m2]
                            n = np.sqrt(np.vdot(_c, _c))
                            if n < self.prec:
                                continue
                            #_c /= n
                            #for old, oldi in zip(res, ind):
                            #    #if oldi[0] != j:
                            #    #    continue
                            #    _c = utils.gram_schmidt(_c, old)
                            #    n = np.sqrt(np.vdot(_c, _c))
                            #    if n < self.prec:
                            #        break
                            #if n < self.prec:
                            #    continue
                            res.append(_c.copy())
                            ind.append([j,mj-j,l,s,m1-self.s1,m2-self.s2])

        res = np.asarray(res)
        ind = np.asarray(ind)
        return res, ind

    def print_all(self, jmax):
        res, ind = self.get_all_ops(jmax)
        if res.size < 1:
            return
        line = "-" * (3*6+5+len(self.allmomenta)*12)
        print("O_s=%d(p_1) x O_s=%d(p_2) -> O(P)_{S,L}^{J,m_J}"%(
                self.s1, self.s2))
        self.print_head(colsize=3*6+5)
        print(line)
        for c, i in zip(res, ind):
            self.print_coeffs_new(c, i)
            print(line)

    def print_old(self, j):
        line = "-" * (3*6+5+len(self.allmomenta)*12)
        print("O_s=%d(p_1) x O_s=%d(p_2) -> O(P)_{S,L}^{J,m_J}"%(
                self.s1, self.s2))
        self.print_head(colsize=3*6+5)
        print(line)
        jmult = int(2*j+1)
        srange = self.s1+self.s2+1
        for mj in range(jmult):
            for s in range(srange):
                for l in range(j+srange):
                    c = self.calc_op(j, mj, l, s)
                    if utils._eq(c):
                        continue
                    #print("j, mj, l, s = %d, %d, %d, %d" % (j, mj-j, l, s))
                    self.print_coeffs(c, [j,mj-j,l,s])
                    print(line)

if __name__ == "__main__":
    print("for checks execute the test script")
