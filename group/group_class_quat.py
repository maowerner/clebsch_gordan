"""Class for groups and their representatives based on quaternions."""

import numpy as np
import itertools as it

import utils
import quat

class TO(object):
    def __init__(self, pref=None):
        self.name = "TO"
        self.npar = 24 # the parameters of the rotations
        self.order = 2 * self.npar # order of the group npar *2 (conjugation)
        
        # set the elements, twice in a row for correct ordering
        self.elements = []
        for el in quat.qPar:
            self.elements.append(quat.QNew.create_from_vector(el, 1))
        for el in quat.qPar:
            self.elements.append(quat.QNew.create_from_vector(-el, 1))
        # select elements when pref is given
        if pref is not None:
            pass

        # set up multiplication table
        self.tmult = np.zeros((self.order, self.order), dtype=int)
        self.make_mult_table()

        # set up list with inverse elements
        self.linv = np.zeros((self.order,), dtype=int)
        self.make_inv_list()

        # determine conjugacy classes
        self.tconjugacy = np.zeros_like(self.tmult, dtype=int)
        self.make_conjugacy_relations()
        self.assign_classes()

        # prepare storage for representations, number of irreps and
        # classes must be the same
        # use register_irrep for adding
        self.irreps = []
        self.irrepsname = []
        self.irrepdim = np.zeros((self.nclasses,), dtype=int)
        self.tchar = np.zeros((self.nclasses, self.nclasses), dtype=complex)

    def make_mult_table(self):
        for i in range(self.order):
            for j in range(self.order):
                res = -999
                tmp = self.elements[i] * self.elements[j]
                for k in range(self.order):
                    if tmp == self.elements[k]:
                        res = k
                        break
                self.tmult[i,j] = res
                if res < 0:
                    print("no element found corresponding to %d*%d" % (i,j))
        self.is_faithful()

    def is_faithful(self):
        # check if complete
        self.faithful = True
        checksum = np.ones((self.order,))*np.sum(range(self.order))
        tmp = np.sum(self.tmult, axis=0)
        if not utils._eq(tmp, checksum):
            self.faithful = False
        tmp = np.sum(self.tmult, axis=1)
        if not utils._eq(tmp, checksum):
            self.faithful = False

    def make_inv_list(self):
        for i in range(self.order):
            for k in range(self.order):
                tmp = self.elements[i] * self.elements[k]
                if tmp == self.elements[0]:
                    self.linv[i] = k
                    break

    def make_conjugacy_relations(self):
        # 2 nested for loops
        for i, j in it.product(range(self.order), repeat=2):
            for k in range(self.order):
                k_inv = self.linv[k]
                j_k = self.tmult[j,k]
                k_inv_j_k = self.tmult[k_inv, j_k]
                if k_inv_j_k == i:
                    self.tconjugacy[i,j] = 1
                    break
    
    def assign_classes(self):
        # assign a class representative for each class
        tmp = np.ones((self.order,), dtype=int) * -1
        for i, j in it.product(range(self.order), repeat=2):
            if self.tconjugacy[i,j] == 1:
                if tmp[j] == -1:
                    tmp[j] = i
        tmp = np.sort(tmp)
        self.nclasses = len(np.unique(tmp))
        # get dimension and representative of each class
        self.cdim = np.zeros((self.nclasses,), dtype=int)
        self.crep = np.zeros((self.nclasses,), dtype=int)
        icls, j = -1, -1
        for i in range(self.order):
            if tmp[i] != j:
                icls += 1
                j = tmp[i]
                self.cdim[icls] = 1
                self.crep[icls] = j
            else:
                self.cdim[icls] += 1
        self.cdimmax = np.max(self.cdim)
        # sort all elements into the classes
        self.lclasses = np.ones((self.nclasses, self.cdimmax), dtype=int) * -1
        for i in range(self.nclasses):
            k = -1
            for j in range(self.order):
                if tmp[j] == self.crep[i]:
                    k += 1
                    self.lclasses[i][k] = j

    def register_irrep(self, irrep):
        pass

    def check_orthogonalities(self):
        # check row orthogonality
        check1 = np.zeros((self.nclasses, self.nclasses), dtype=complex)
        for i in range(self.nclasses):
            for j in range(self.nclasses):
                res = 0.
                for k in range(self.nclasses):
                    res += self.tchar[i,k] * self.tchar[j,k].conj() * self.cdim[k]
                check1[i,j] = res
        tmp = self.order * np.identity(self.nclasses)
        t1 = utils._eq(check1, tmp)
        # check row orthogonality
        check2 = np.zeros((self.nclasses, self.nclasses), dtype=complex)
        for i in range(self.nclasses):
            for j in range(self.nclasses):
                res = 0.
                for k in range(self.nclasses):
                    res += self.tchar[k,i] * self.tchar[k,j].conj()
                check2[i,j] = res
        tmp = np.diag(np.ones((self.nclasses,))*self.order/np.asarray(self.cdim))
        t2 = utils._eq(check2, tmp)
        return t1, t2

    def characters_of_SU2(self, j, inv=False):
        """Character of SU(2) with multiplicity [j] = 2*j+1 for
        all conjugacy classes.

        Parameter
        ---------
        j : int
            The multiplicity of the angular momentum.

        Returns
        -------
        char : float
            The character of SU(2).
        """
        omegas = np.asarray([2.*np.arccos(self.elements[x].q[0]) for x in self.crep])
        _inv = np.asarray([self.elements[x].i for x in self.crep])
        _sum = np.zeros_like(omegas)
        if j%2 == 0: # half-integer spin
            n = j//2
            for k in range(1, n+1):
                _sum += np.cos(0.5 * (2*k-1) * omegas)
        else:
            n = (j-1)//2
            _sum += 0.5
            for k in range(1, n+1):
                _sum += np.cos(k * omegas)
        _sum *= 2.
        if inv:
            _sum *= _inv
        return _sum

    def print_mult_table(self):
        print("")
        tmpstr = [" %3d" % x for x in range(self.order)]
        tmpstr = "|".join(tmpstr)
        tmpstr = "".join(["    |", tmpstr])
        print(tmpstr)
        print("_".center(self.order*4, "_"))
        for i in range(self.order):
            tmpstr = [" %3d" % x for x in self.tmult[i]]
            tmpstr = "|".join(tmpstr)
            tmpstr = "".join([" %3d|" % i, tmpstr])
            print(tmpstr)

    def print_class_members(self):
        for i in range(self.nclasses):
            tmpstr = ["%d" % x for x in self.lclasses[i] if x != -1]
            tmpstr = ", ".join(tmpstr)
            print("class %2d: %s" % (i, tmpstr))

if __name__ == "__main__":
    print("for checks execute the test script")


