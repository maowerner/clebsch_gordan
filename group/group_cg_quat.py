"""Class for the clebsch-gordan coefficients of a group."""

import numpy as np
import itertools as it

import group_class
import utils

class TOhCG(object):
    def __init__(self, p, p1, p2, groups=None):
        """p, p1, and p2 are the magnitudes of the momenta.
        """
        self.prec = 1e-6
        # save the norm of the momenta for the combined system
        # and each particle
        self.p = p
        self.p1 = p1
        self.p2 = p2
        # lookup table for reference momenta
        lpref = [np.asarray([0.,0.,0.]), np.asarray([0.,0.,1.]), 
                 np.asarray([1.,1.,0.]), np.asarray([1.,1.,1.])]
        # save reference momenta
        self.pref = lpref[p]
        self.pref1 = lpref[p1]
        self.pref2 = lpref[p2]

        # get the basic groups
        if groups is None:
            self.g0 = None
            self.g = None
            self.g1 = None
            self.g2 = None
        else:
            self.g0 = groups[0]
            self.g = groups[p]
            self.g1 = groups[p1]
            self.g2 = groups[p2]

        # get the cosets, always in the maximal group (2O here)
        # returns None if groups is None
        self.coset1 = self.gen_coset(groups, self.p1)
        self.coset2 = self.gen_coset(groups, self.p2)
        #print(self.coset1)
        #print(self.coset2)

        # generate the allowed momentum combinations and sort them into cosets
        self.gen_momenta()

        # calculate induced rep gamma
        # here for p1 and p2 the A1(A2) irreps are hard-coded
        # since only these contribute to pi-pi scattering
        if groups is None:
            self.gamma1 = None
            self.gamma2 = None
            self.dim1 = 0
            self.dim2 = 0
        else:
            irstr = "A1u" if int(p1) in [0,3] else "A2u"
            #irstr = "E1g"
            self.gamma1 = self.gen_ind_reps(groups, p1, irstr, self.coset1)
            self.dim1 = 1
            irstr = "A1u" if int(p2) in [0,3] else "A2u"
            #irstr = "E1g"
            self.gamma2 = self.gen_ind_reps(groups, p2, irstr, self.coset2)
            self.dim2 = 1
            self.sort_momenta(groups[0])
        #print(self.gamma1[:5])
        #print("traces of induced representation")
        #print(self.spur1)
        #print(self.spur2)

        self.irreps = []
        self.cgs = []

        self.calc_cg_new()

    def gen_coset(self, groups, p):
        """Cosets contain the numbers of the rotation objects
        """
        if groups is None:
            return None
        g0 = groups[0]
        g1 = groups[p]
        n = int(g0.order/g1.order)
        if n == 0:
            raise RuntimeError("number of cosets is 0!")
        coset = np.zeros((n, g1.order), dtype=int)
        l = g0.order
        l1 = g1.order
        # set the subgroup as first coset
        count = 0
        for r in range(l1):
            elem = g1.lelements[r]
            if elem in g0.lelements:
                coset[0, count] = elem
                count += 1
        # calc the cosets
        uniq = np.unique(coset)
        cnum = 1 # coset number
        for elem in g0.lelements:
            if elem in uniq:
                continue
            count = 0
            for elem1 in g1.lelements:
                if elem1 in g0.lelements:
                    look = g0.lelements.index(elem1)
                    el = g0.tmult_global[elem, look]
                    coset[cnum, count] = el
                    count += 1
            cnum += 1
            uniq = np.unique(coset)
        if len(uniq) != g0.order:
            print("some elements got lost!")
        if cnum != n:
            print("some coset not found!")
        return coset

    def gen_momenta(self):
        pm = 4 # maximum component in each direction
        def _abs(x):
            return np.dot(x, x) <= pm
        def _abs1(x,a):
            return np.dot(x, x) == a
        gen = it.ifilter(_abs, it.product(range(-pm,pm+1), repeat=3))
        lp3 = [np.asarray(y, dtype=int) for y in gen]
        self.momenta = [y for y in it.ifilter(lambda x: _abs1(x,self.p), lp3)]
        self.momenta1 = [y for y in it.ifilter(lambda x: _abs1(x,self.p1), lp3)]
        self.momenta2 = [y for y in it.ifilter(lambda x: _abs1(x,self.p2), lp3)]
        

    def sort_momenta(self, g0):
        # check if cosets exists
        if self.coset1 is None or self.coset2 is None:
            self.smomenta1 = None
            self.smomenta2 = None
            return
        def check_coset(g0, pref, p, coset):
            res = []
            for elem in coset:
                look = g0.lelements.index(elem)
                quat = g0.elements[look]
                rvec = quat.rotation_matrix(True).dot(pref)
                c1 = utils._eq(rvec, p)
                if c1:
                    res.append(True)
                else:
                    res.append(False)
            return res
        # search for conjugacy class so that
        # R*p_ref = p
        res1 = []
        res2 = []
        for p1 in self.momenta1:
            for i,c in enumerate(self.coset1):
                t = check_coset(g0, self.pref1, p1, c)
                if np.all(t):
                    res1.append((p1, i))
                    break
        for p2 in self.momenta2:
            for i,c in enumerate(self.coset2):
                t = check_coset(g0, self.pref2, p2, c)
                if np.all(t):
                    res2.append((p2, i))
                    break
        self.smomenta1 = res1
        if len(self.smomenta1) != len(self.momenta1):
            print("some vectors not sorted, momentum 1")
        self.smomenta2 = res2
        if len(self.smomenta2) != len(self.momenta2):
            print("some vectors not sorted, momentum 2")
        # generate lists with all allowed momenta combinations and their
        # combined coset+dim indices
        dimcomb = [x for x in it.product(range(self.dim1), range(self.dim2))]
        self.allmomenta = []
        self.indices = []
        for p in self.momenta:
            for p1 in self.momenta1:
                for p2 in self.momenta2:
                    if utils._eq(p1+p2-p):
                        self.allmomenta.append((p, p1, p2))
                        j1, j2 = self.check_all_cosets(p, p1, p2)
                        for x in dimcomb:
                            self.indices.append((j1*self.dim1+x[0], j2*self.dim2+x[1]))


    def gen_ind_reps(self, groups, p, irstr, coset):
        g0 = groups[0]
        g1 = groups[p]
        ir = g1.irreps[g1.irrepsname.index(irstr)]
        dim = ir.dim
        ndim = (g0.order, coset.shape[0]*dim, coset.shape[0]*dim)
        gamma = np.zeros(ndim, dtype=complex)
        for ind, r in enumerate(g0.lelements):
            for cj, rj in enumerate(coset[:,0]):
                rrj = g0.tmult[r, rj]
                indj = slice(cj*dim, (cj+1)*dim)
                for ci, ri in enumerate(coset[:,0]):
                    riinv = g0.linv[ri]
                    riinvrrj = g0.tmult[riinv, rrj]
                    indi = slice(ci*dim, (ci+1)*dim)
                    if riinvrrj not in coset[0]:
                        continue
                    elem = g1.lelements.index(riinvrrj)
                    gamma[ind, indi, indj] = ir.mx[elem]
        return gamma


    def check_all_cosets(self, p, p1, p2):
        j1, j2 = None, None
        for m, j in self.smomenta1:
            if utils._eq(p1,m):
                j1 = j
                break
        if j1 is None:
            print("j1 is None")
        for m, j in self.smomenta2:
            if utils._eq(p2,m):
                j2 = j
                break
        if j2 is None:
            print("j2 is None")
        return j1, j2

    def multiplicities(self):
        multi = np.zeros((self.g.nclasses,), dtype=complex)
        for i in range(self.g.order):
            chars = np.asarray([np.trace(ir.mx[i]).conj() for ir in self.g.irreps])
            look = self.g.lelements[i]
            tr1 = tr2 = 0.
            for i1, i2 in self.indices:
                if i1 != i2:
                    continue
                tr1 += self.gamma1[look,i1,i1]
                tr2 += self.gamma2[look,i2,i2]
            chars *= tr1*tr2
            #chars *= np.trace(self.gamma1[look])
            #chars *= np.trace(self.gamma2[look])
            multi += chars
        #multi = np.real_if_close(np.rint(multi/self.g.order))
        multi = np.real_if_close(multi/self.g.order)
        return multi

    def check_index(self, mu1, mu2):
        i1 = mu1 // self.dim1
        p1 = None
        for m, j in self.smomenta1:
            if j == i1:
                p1 = m
                break
        if p1 is None:
            raise RuntimeError("Momentum 1 not found")

        i2 = mu2 // self.dim2
        p2 = None
        for m, j in self.smomenta2:
            if j == i2:
                p2 = m
                break
        if p2 is None:
            raise RuntimeError("Momentum 2 not found")
        
        p12 = p1+p2
        for m in self.momenta:
            if utils._eq(p12, m):
                return True
        return False

    def calc_cg_new(self):
        multi = np.zeros((self.g.nclasses,), dtype=int)
        dim1 = self.gamma1.shape[1]
        dim2 = self.gamma2.shape[1]
        dim12 = dim1*dim2
        coeff = np.zeros((len(self.indices),), dtype=complex)
        self.cgnames = []
        self.cgind = []
        self.cg = []
        lind = []
        for indir, ir in enumerate(self.g.irreps):
            lcoeffs = []
            dim = ir.dim
            # loop over all column index combinations that conserve the COM momentum
            for mup, (mu1, mu2) in it.product(range(dim), self.indices):
            #for mup, mu1, mu2 in it.product(range(dim), range(dim1), range(dim2)):
                if not self.check_index(mu1, mu2):
                    continue
                # loop over the row of the final irrep
                for mu in range(dim):
                    coeff.fill(0.)
                    # loop over all combinations of rows of the induced
                    # representations
                    for ind1, (mu1p, mu2p) in enumerate(self.indices):
                    #for mu1p, mu2p in it.product(range(dim1), range(dim2)):
                        if not self.check_index(mu1p, mu2p):
                            continue
                        ind12 = dim2 * mu1p + mu2p
                        co = 0.j
                        for i in range(self.g.order):
                            tmp = ir.mx[i][mu, mup].conj()
                            look = self.g.lelements[i]
                            tmp *= self.gamma1[look, mu1p, mu1]
                            tmp *= self.gamma2[look, mu2p, mu2]
                            co += tmp
                        coeff[ind1] = co*dim
                        #coeff[ind12] = co*dim
                    coeff /= self.g.order
                    ncoeff = np.sqrt(np.vdot(coeff, coeff))
                    # if norm is 0, try next combination of mu', mu1, mu2
                    if ncoeff < self.prec:
                        continue
                    else:
                        coeff /= ncoeff
                    # orthogonalize w.r.t. already found vectors of same irrep
                    for vec in lcoeffs:
                        coeff = utils.gram_schmidt(coeff, vec, prec=self.prec)
                        ncoeff = np.sqrt(np.vdot(coeff, coeff))
                        # if zero vector, try next combination of mu', mu1, mu2
                        if ncoeff < self.prec:
                            break
                    if ncoeff < self.prec:
                        continue
                    # orthogonalize w.r.t. already found vectors of other irreps
                    for lcg in self.cg:
                        #print("checking coeff against:")
                        #print(lcg)
                        for vec in lcg:
                            coeff = utils.gram_schmidt(coeff, vec, prec=self.prec)
                            ncoeff = np.sqrt(np.vdot(coeff, coeff))
                            # if zero vector, try next combination of mu', mu1, mu2
                            if ncoeff < self.prec:
                                break
                        if ncoeff < self.prec:
                            break
                    if ncoeff > self.prec:
                        lcoeffs.append(coeff.copy())
                        lind.append((mu, mup, mu1, mu2))
                        multi[indir] += 1
            if multi[indir] > 0:
                print("%s: %d times" % (ir.name, multi[indir]/dim))
                self.cgnames.append((ir.name, multi[indir]/dim, dim))
                self.cg.append(np.asarray(lcoeffs).copy())
                #print("appended coeffs:")
                #print(self.cg[-1])
                self.cgind.append(np.asarray(lind).copy())

    def display(self, emptyline=None):
        def tostring(data):
            tmp = ["%+.3f%+.3fj" % (x.real, x.imag) for x in data]
            tmp = ",".join(tmp)
            tmp = "".join(("(", tmp, ")"))
            return tmp
        def momtostring(data):
            tmp = []
            for d in data:
                tmp.append("[%+d,%+d,%+d]" % (d[0], d[1], d[2]))
            tmpstr = "%s x %s -> %s" % (tmp[1], tmp[2], tmp[0])
            return tmpstr

        count = 0
        dim1 = self.gamma1.shape[1]
        dim2 = self.gamma2.shape[1]
        # loop over irreps
        for i, (name, multi, dim) in enumerate(self.cgnames):
            if multi < 1:
                continue
            print(" %s ".center(20,"*") % name)
            # loop over multiplicities
            for m in range(multi):
                print("multiplicity %d" % m)
                select = slice(m*dim, (m+1)*dim)
                # loop over momenta
                print("full vector")
                print(self.cg[i][select])
                for ind, (p, p1, p2) in enumerate(self.allmomenta):
                    #i1, i2 = self.check_all_cosets(p, p1, p2)
                    #ind = i1*dim2 + i2
                    data = self.cg[i][select,ind]
                    tmpstr = "%s: %s" % (momtostring([p,p1,p2]), tostring(data))
                    print(tmpstr)
                    if emptyline is not None:
                        count += 1
                        if count == emptyline:
                            print("")
                            count = 0

if __name__ == "__main__":
    print("for checks execute the test script")
