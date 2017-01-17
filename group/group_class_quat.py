"""Class for groups and their representatives based on quaternions."""

import numpy as np
import itertools as it
from timeit import default_timer as timer

import utils
import quat
import group_generators_quat as gg
from rotations import mapping

class TOh(object):
    def __init__(self, pref=None, withinversion=True, debug=0, irreps=False):
        if not withinversion:
            raise RuntimeError("only double cover octahedral group works!")
        self.name = "TO"
        self.npar = 24 # the parameters of the rotations
        self.pref = pref
        self.withinversion = withinversion
        self.debug = debug
        
        # set the elements
        # defines self.elements, self.lelements
        # see comment in select_elements
        clockalls = timer()
        clock1s = timer()
        self.select_elements()
        self.order = len(self.elements) # order of the group
        clock1e = timer()
        if debug > 1:
            print("element selection: %.2fs" % (clock1e - clock1s))

        # set up multiplication table
        # defines self.faithful
        clock1s = timer()
        self.tmult = np.zeros((self.order, self.order), dtype=int)
        self.tmult_global = np.zeros((self.order, self.order), dtype=int)
        self.make_mult_table()
        clock1e = timer()
        if debug > 1:
            print("multiplication table: %.2fs" % (clock1e - clock1s))

        # set up list with inverse elements
        clock1s = timer()
        self.linv = np.zeros((self.order,), dtype=int)
        self.linv_global = np.zeros((self.order,), dtype=int)
        self.make_inv_list()
        clock1e = timer()
        if debug > 1:
            print("inverse elements: %.2fs" % (clock1e - clock1s))

        # determine conjugacy classes
        # defines self.nclasses, self.cdim, self.crep,
        # self.cdimmax, self.lclasses
        clock1s = timer()
        self.tconjugacy = np.zeros_like(self.tmult, dtype=int)
        self.make_conjugacy_relations()
        self.assign_classes()
        clock1e = timer()
        if debug > 1:
            print("conjugacy classes: %.2fs" % (clock1e - clock1s))

        # prepare storage for representations, number of irreps and
        # classes must be the same
        # use register_irrep for adding
        clock1s = timer()
        self.irreps = []
        self.irrepsname = []
        self.irrepdim = np.zeros((self.nclasses,), dtype=int)
        self.tchar = np.zeros((self.nclasses, self.nclasses), dtype=complex)
        if irreps:
            self.find_irreps()
        clock1e = timer()
        clockalle = timer()
        if debug > 1:
            print("irrep selection: %.2fs" % (clock1e - clock1s))
            print("total time: %.2fs" % (clockalle - clockalls))

    def select_elements(self):
        # self.elements contains the quaternions
        # self.lelements contains the "global" (unique) index of the element,
        # making the elements comparable between different groups
        self.elements = []
        self.lelements = []
        # all possible elements for the double cover octahedral group
        for i, el in enumerate(quat.qPar):
            self.elements.append(quat.QNew.create_from_vector(el, 1))
            self.lelements.append(i);
        if self.withinversion:
            for i, el in enumerate(quat.qPar):
                self.elements.append(quat.QNew.create_from_vector(el, -1))
                self.lelements.append(i+24);
        for i, el in enumerate(quat.qPar):
            self.elements.append(quat.QNew.create_from_vector(-el, 1))
            self.lelements.append(i+48);
        if self.withinversion:
            for i, el in enumerate(quat.qPar):
                self.elements.append(quat.QNew.create_from_vector(-el, -1))
                self.lelements.append(i+72);
        self.p2 = 0
        # select elements when pref is given
        if self.pref is not None:
            self.p2 = np.vdot(self.pref, self.pref)
            selected = []
            elem = []
            for el, num in zip(self.elements, self.lelements):
                tmp = el.rotation_matrix(self.withinversion).dot(self.pref)
                c1 = utils._eq(tmp - self.pref)
                if c1:
                    selected.append(num)
                    elem.append(el)
            if self.debug > 0:
                print("The group with P_ref = %r has %d elements:" % (
                        self.pref.__str__(), len(elem)))
                tmpstr = ["%d" % x for x in selected]
                tmpstr = ", ".join(tmpstr)
                print("[%s]" % tmpstr)
            self.elements = elem
            self.lelements = selected

    def make_mult_table(self):
        for i in range(self.order):
            for j in range(self.order):
                res = -999
                tmp = self.elements[i] * self.elements[j]
                for k in range(self.order):
                    if tmp == self.elements[k]:
                        res = k
                        break
                if res < 0:
                    print("no element found corresponding to %d*%d" % (i,j))
                self.tmult[i,j] = res
                self.tmult_global[i,j] = self.lelements[res]
        self.is_faithful()

    def is_faithful(self):
        # check if complete
        self.faithful = True
        checksum = np.ones((self.order,))*np.sum(self.lelements)
        u, ind = np.unique(self.tmult, return_index=True)
        if len(u) != self.order:
            self.faithful = False
        tmp = np.sum(self.tmult_global, axis=0)
        if not utils._eq(tmp, checksum):
            self.faithful = False
        tmp = np.sum(self.tmult_global, axis=1)
        if not utils._eq(tmp, checksum):
            self.faithful = False

    def make_inv_list(self):
        for i in range(self.order):
            for k in range(self.order):
                tmp = self.elements[i] * self.elements[k]
                if tmp == self.elements[0]:
                    self.linv[i] = k
                    self.linv_global[i] = self.lelements[k]
                    break

    def make_conjugacy_relations(self):
        # 2 nested for loops
        for i, j in it.product(range(self.order), repeat=2):
            #print("indices %d, %d" % (i,j))
            for k in range(self.order):
                k_inv = self.linv[k]
                j_k = self.tmult[j,k]
                k_inv_j_k = self.tmult[k_inv, j_k]
                #print("k^-1 (%d) * (j (%d) * k (%d)) (%d) = %d (check %d)" % (
                #        k_inv, j, k, j_k, k_inv_j_k, i))
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
        tmps = np.sort(tmp)
        self.nclasses = len(np.unique(tmp))
        # get dimension and representative of each class
        self.cdim = np.zeros((self.nclasses,), dtype=int)
        self.crep = np.zeros((self.nclasses,), dtype=int)
        icls, j = -1, -1
        for i in range(self.order):
            if tmps[i] != j:
                icls += 1
                j = tmps[i]
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

    def check_ortho(self, irrep):
        if len(self.irreps) == 0:
            return True
        char = irrep.characters(self.crep)
        n = len(self.irreps)
        # check row orthogonality
        check1 = np.zeros((n,), dtype=complex)
        for i in range(n):
            res = 0.
            for k in range(self.nclasses):
                res += self.tchar[i,k] * char[k].conj() * self.cdim[k]
            check1[i] = res
        t1 = utils._eq(check1)
        return t1

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
        # check column orthogonality
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

    def characters_of_SU2(self, j, useinv=True):
        """Character of SU(2) with multiplicity [j] = 2*j+1 for
        all conjugacy classes.

        Parameter
        ---------
        j : int
            The multiplicity of the angular momentum.

        Returns
        -------
        char : ndarray
            The characters of SU(2).
        """
        omegas = np.asarray([2.*np.arccos(self.elements[x].q[0]) for x in self.crep])
        _inv = np.asarray([float(self.elements[x].i) for x in self.crep])
        _char = np.zeros_like(omegas)
        if j%2 == 0: # half-integer spin
            n = j//2
            for k in range(1, n+1):
                _char += np.cos(0.5 * (2*k-1) * omegas)
        else:
            n = (j-1)//2
            _char += 0.5
            for k in range(1, n+1):
                _char += np.cos(k * omegas)
        _char *= 2.
        if useinv:
            _char *= _inv
        return _char

    def multiplicity_of_SU2(self, j, useinv=True):
        """Multiplicites of irreps for SU(2) with multiplicity [j] = 2*j+1.

        Parameter
        ---------
        j : int
            The multiplicity of the angular momentum.
        useinv : bool
            Use the inversion flag of the representations.

        Returns
        -------
        multi : ndarray
            The multiplicities of the irreps for the SU(2) representation.
        """
        _char = self.characters_of_SU2(j, useinv=useinv)
        multi = self.tchar.dot(_char*self.cdim)
        multi = np.real_if_close(np.rint(multi/float(self.order)))
        return multi

    def print_mult_table(self):
        print("multiplication table\n")
        n = int(self.order)/int(self.npar)
        line = "_".center(self.npar*5, "_")
        for n1 in range(n):
            head = ["%2d" % (x+n1*self.npar) for x in range(self.npar)]
            head = " ".join(head)
            head = "".join(["\n   [", head, "]"])
            for n2 in range(n):
                print(head)
                print(line)
                for i in range(self.npar):
                    tmpstr = ["%2d" % x for x in self.tmult[i+n2*self.npar,n1*self.npar:(n1+1)*self.npar]]
                    tmpstr = " ".join(tmpstr)
                    tmpstr = "".join(["%2d | [" % (i+n2*self.npar), tmpstr, "]"])
                    print(tmpstr)

    def print_char_table(self):
        def _tostring(x):
            if np.isclose(x.imag, 0.):
                return " %6.3f " % x.real
            elif np.isclose(x.real, 0.):
                return " %6.3fJ" % x.imag
            else:
                return " %6.3f+" % x.real
        print("")
        tmpstr = [" %6d " % x for x in range(self.nclasses)]
        tmpstr = "|".join(tmpstr)
        tmpstr = "".join(["    |", tmpstr])
        print(tmpstr)
        print("_".center(self.nclasses*9+3, "_"))
        for i in range(self.nclasses):
            tmpstr = [_tostring(x) for x in self.tchar[i]]
            tmpstr = "|".join(tmpstr)
            try:
                tmpstr = "".join([" %3s|" % self.irreps[i].name, tmpstr])
            except IndexError:
                tmpstr = "".join([" %3d|" % i, tmpstr])
            print(tmpstr)

    def print_class_members(self):
        for i in range(self.nclasses):
            tmpstr = ["%d" % x for x in self.lclasses[i] if x != -1]
            tmpstr = ", ".join(tmpstr)
            print("class %2d: %s" % (i, tmpstr))

    def find_irreps(self):
        # find the possible combinations of irrep dimensions
        self.find_possible_dims()
        # get 1D irreps
        self.find_1D()
        alldone = self.check_possible_dims()
        if not alldone:
            for d in range(2, 5):
                self.get_irreps(d)
                alldone = self.check_possible_dims()
                if alldone:
                    break
        # find special cases, sorted by time needed from fast to slow
        if not alldone:
            for d in range(2, 3):
                self.get_irreps_special(d)
                alldone = self.check_possible_dims()
                if alldone:
                    break
        if not alldone:
            self.find_1D_special()
            alldone = self.check_possible_dims()
        #self.print_char_table()
        if not alldone:
            msg = "did not find all irreps, found %d/%d" % (len(self.irreps), self.nclasses)
            print(msg)
            #raise RuntimeError(msg)
        else:
            check1, check2 = self.check_orthogonalities()
            if not check1 or not check2:
                print("row orthogonality: %r\ncolumn orthogonality: %r" % (
                        check1, check2))

    def find_possible_dims(self):
        def _op(data):
            # check if the sum of squares is equal to the order
            check1 = (np.sum(np.square(data)) == self.order)
            return check1
        nmax = int(np.floor(np.sqrt(self.order)))
        # generator for all combinations of dimensions, sorted
        gen = it.combinations_with_replacement(range(1,nmax+1), self.nclasses)
        # filter all combinations which satisfy function
        # _op defined above
        tmp = np.asarray(list(it.ifilter(_op, gen)), dtype=int)
        # generate array with frequency of dimension for every possible
        # combination of dimensions
        self.pos = np.zeros((len(tmp), nmax), dtype=int)
        for n in range(nmax):
            self.pos[:,n] = np.sum(tmp == n+1, axis=1)

    def check_possible_dims(self):
        if len(self.irrepsname) == 0:
            return False
        elif len(self.irreps) == self.nclasses:
            return True
        elif self.pos is None:
            return False
        nmax = self.pos.shape[1]
        fre = np.zeros((nmax,), dtype=int)
        for n in range(nmax):
            fre[n] = np.sum(self.irrepdim == n+1)
        skip = False
        tmp = []
        for p in self.pos:
            skip = False
            for n in range(nmax):
                if not (fre[n] <= p[n]):
                    skip = True
                    break
            if not skip:
                tmp.append(p)
        tmp = np.asarray(tmp, dtype=int)
        if tmp.size == nmax:
            self.checkdims = tmp
            self.pos = None
        elif tmp.size == 0:
            raise RuntimeError("no possible dimensions found")
        else:
            self.pos = tmp.copy()
        return False

    def append_irrep(self, irrep):
        self.irreps.append(irrep)
        self.irrepsname.append(irrep.name)
        ind = len(self.irreps)-1
        self.irrepdim[ind] = irrep.dim
        irrep.irid = ind
        self.tchar[ind] = irrep.characters(self.crep)

    def find_1D(self):
        ir = TOh1D(self.elements)
        ir.name = "A1g"
        if ir.is_representation(self.tmult):
            self.append_irrep(ir)
        self.flip = np.asarray([x for x in self.flip_reps(ir)], dtype=int)
        # temporary suffixes
        self.suffixes = ["%dx"%x for x in range(2,len(self.flip)+2)]
        # create the correct suffixes
        tmpu = np.asarray([self.elements[x].i for x in self.crep], dtype=int)
        for i, f in enumerate(self.flip):
            if np.allclose(f, tmpu):
                self.suffixes[i] = "1u"
                break
        count = 2
        for i in range(len(self.suffixes)):
            if not self.suffixes[i].endswith("x"):
                continue
            self.suffixes[i] = "%dg" % count
            tmpi = tmpu * self.flip[i]
            ind = 0
            for j, f in enumerate(self.flip):
                if np.allclose(tmpi, f):
                    ind = j
                    break
            self.suffixes[ind] = "%du" % count
            count += 1
        # sort the vectors
        tmp = ["1u"]
        for i in range(2,count):
            tmp.append("%dg" % i)
            tmp.append("%du" % i)
        tmpf = []
        for s in tmp:
            tmpf.append(self.flip[self.suffixes.index(s)])
        self.flip = np.asarray(tmpf)
        self.suffixes = tmp
        for f, s in zip(self.flip, self.suffixes):
            ir = TOh1D(self.elements)
            ir.flip_classes(f, self.lclasses)
            ir.name = "".join(("A", s))
            if ir.is_representation(self.tmult) and self.check_ortho(ir):
                self.append_irrep(ir)

    def find_1D_special(self):
        ir = TOh1D(self.elements)
        self.flip_i = [x for x in self.flip_reps_imaginary(ir)]
        self.suffixes = ["%d"%x for x in range(1,10)] # g/u for even odd
        #print("possible vectors: %d" % len(self.flip_i))
        for f, s in zip(self.flip_i, self.suffixes):
            ir = TOh1D(self.elements)
            ir.flip_classes(f, self.lclasses)
            ir.name = "".join(("K", s))
            if ir.is_representation(self.tmult) and self.check_ortho(ir):
                self.append_irrep(ir)

    def flip_reps(self, irrep):
        n = self.nclasses
        mx_backup = irrep.mx.copy()
        # flip classes
        for k in range(1, n):
            #print("flip %d classes" % k)
            for ind in it.combinations(range(1,n), k):
                fvec = np.ones((n,))
                for x in ind:
                    fvec[x] *= -1
                irrep.flip_classes(fvec, self.lclasses)
                check1 = np.sum(irrep.mx)
                check2 = irrep.is_representation(self.tmult)
                irrep.mx = mx_backup.copy()
                if utils._eq(check1) and check2:
                    yield fvec.copy()

    def flip_reps_imaginary(self, irrep):
        def check_index(i1, i2):
            for x in i2:
                try:
                    i1.index(x)
                    return True
                except ValueError:
                    continue
            return False
        def f_vec(n, ind1, ind2, indm):
            fvec = np.ones((n,), dtype=complex)
            for i in ind1:
                fvec[i] *= 1.j
            for i in ind2:
                fvec[i] *= -1.j
            for i in indm:
                fvec[i] *= -1.
            return fvec

        n = self.nclasses
        mx_backup = irrep.mx.copy()
        count = 0
        # multiply classes with -1, i and -i,
        # always the same number of classes with +i and -i
        # total number of classes flipped
        for kt in range(3, n):
            #print("flip %d classes" % (kt))
            # half the number of classes with imaginary flip
            for ki in range(1, kt//2):
                # get indices for classes with +/- i
                for ind1 in it.combinations(range(1, n), ki):
                    for ind2 in it.combinations(range(1,n), ki):
                        # check if some class is in both index arrays
                        # and skip if it is
                        if check_index(ind1, ind2):
                            #print("skipping %r and %r" % (ind1, ind2))
                            continue
                        # get indices for classes with -1
                        for indm in it.combinations(range(1,n), kt-2*ki):
                            # check if some class is already taken
                            # and skip if it is
                            if check_index(ind1, indm):
                                #print("skipping %r and %r" % (ind1, ind2))
                                continue
                            # check if some class is already taken
                            # and skip if it is
                            if check_index(ind2, indm):
                                #print("skipping %r and %r" % (ind1, ind2))
                                continue
                            #print("using %r and %r" % (ind1, ind2))
                            fvec = f_vec(n, ind1, ind2, indm)
                            irrep.flip_classes(fvec, self.lclasses)
                            #print(irrep.characters(self.crep))
                            check1 = np.sum(irrep.mx)
                            check2 = irrep.is_representation(self.tmult)
                            irrep.mx = mx_backup.copy()
                            #print("%r, %r, %r: %r, %r" % (ind1, ind2, indm,
                            #        check1, check2))
                            if utils._eq(check1) and check2:
                                #print("yield")
                                count += 1
                                yield fvec.copy()
                                # hard-coded due to long runtime
                                if count == 4:
                                    return

    def get_irreps(self, dim, naming=None):
        if dim == 2:
            rep = lambda x=None: TOh2D(self.elements)
            _name = "E" if (naming is None) else naming
        elif dim == 3:
            rep = lambda x=None: TOh3D(self.elements)
            _name = "T" if (naming is None) else naming
        elif dim == 4:
            rep = lambda x=None: TOh4D(self.elements)
            _name = "G" if (naming is None) else naming
        elif dim == 1:
            raise RuntimeError("there is a special routine for 1D irreps")
        else:
            raise RuntimeError("%dD irreps not implemented" % dim)

        if self.debug > 2:
            print("finding %dD irreps" % dim)
        ir = rep()
        ir.name = "".join([_name, "1g"])
        check1 = ir.is_representation(self.tmult)
        check2 = self.check_ortho(ir)
        if check1 and check2:
            self.append_irrep(ir)
        if self.debug > 2:
            print("Irrep %s is representation (%r) and orthogonal (%r)" % (
                    ir.name, check1, check2))
        for f, s in zip(self.flip, self.suffixes):
            ir = rep()
            ir.flip_classes(f, self.lclasses)
            ir.name = "".join([_name, s])
            check1 = ir.is_representation(self.tmult)
            check2 = self.check_ortho(ir)
            if check1 and check2:
                self.append_irrep(ir)
            if self.debug > 2:
                print("Irrep %s is representation (%r) and orthogonal (%r)" % (
                        ir.name, check1, check2))

    def get_irreps_special(self, dim, naming=None):
        if dim == 2:
            rep = lambda x=None: TOh2Dp(self.elements, self.pref)
            _name = "Ep" if (naming is None) else naming
        elif dim == 1:
            raise RuntimeError("there is a special routine for 1D irreps")
        else:
            raise RuntimeError("%dD irreps not implemented" % dim)

        if self.debug > 2:
            print("finding special %dD irreps" % dim)
        ir = rep()
        ir.name = "".join([_name, "1g"])
        check1 = ir.is_representation(self.tmult)
        check2 = self.check_ortho(ir)
        if check1 and check2:
            self.append_irrep(ir)
        if self.debug > 2:
            print("Irrep %s is representation (%r) and orthogonal (%r)" % (
                    ir.name, check1, check2))
        for f, s in zip(self.flip, self.suffixes):
            ir = rep()
            ir.flip_classes(f, self.lclasses)
            ir.name = "".join([_name, s])
            check1 = ir.is_representation(self.tmult)
            check2 = self.check_ortho(ir)
            if check1 and check2:
                self.append_irrep(ir)
            if self.debug > 2:
                print("Irrep %s is representation (%r) and orthogonal (%r)" % (
                        ir.name, check1, check2))

class TOhRep(object):
    def __init__(self, dimension):
        self.dim = dimension
        self.irid = -1
        self.name = " "
        self.mx = None
        self.char = None
        self.rep = None

    def characters(self, representatives):
        if self.mx is None:
            return np.nan
        elif ((self.char is not None) and
              (self.rep is not None) and
              (utils._eq(self.rep, representatives))):
            return self.char
        else:
            char = np.zeros((len(representatives),), dtype=complex)
            for i, r in enumerate(representatives):
                char[i] = np.trace(self.mx[r])
            self.char = char
            self.rep = representatives
            return char

    def is_representation(self, tmult, verbose=False):
        n = self.mx.shape[0]
        for i in range(n):
            mxi = self.mx[i]
            for j in range(n):
                mxj = self.mx[j]
                mxk = self.mx[tmult[i,j]]
                mxij = mxi.dot(mxj)
                if not utils._eq(mxij, mxk):
                    if verbose:
                        print("elements %d * %d (%r * %r) not the same as %d (%r / %r)" % (i, j, mxi, mxj, tmult[i,j], mxij, mxk))
                    return False
        return True

    def flip_classes(self, vec, classes):
        """multiply elements by contents of vec based on the class
        they are in."""
        if self.char is not None:
            self.char = None
        for v, c in zip(vec, classes):
            for el in c:
                if el == -1:
                    break
                self.mx[el] *= v

class TOh1D(TOhRep):
    def __init__(self, elements):
        TOhRep.__init__(self, 1)
        self.name = "TOh1D"
        self.mx = gg.genJ0(elements)

class TOh2D(TOhRep):
    def __init__(self, elements):
        TOhRep.__init__(self, 2)
        self.name = "TOh2D"
        self.mx = gg.genJ1_2(elements)

class TOh3D(TOhRep):
    def __init__(self, elements):
        TOhRep.__init__(self, 3)
        self.name = "TOh3D"
        self.mx = gg.genT1CMF(elements)
        #self.mx = gg.genJ1(elements)

class TOh4D(TOhRep):
    def __init__(self, elements):
        TOhRep.__init__(self, 4)
        self.name = "TOh4D"
        self.mx = gg.genJ3_2(elements)

class TOh2Dp(TOhRep):
    def __init__(self, elements, pref=None):
        TOhRep.__init__(self, 2)
        self.name = "TOh2Dp"
        p2 = 0 if (pref is None) else np.dot(pref, pref)
        if p2 in [0, 3]:
            self.mx = gg.genEpCMF(elements)
        elif p2 == 1:
            self.mx = gg.genEpMF1(elements)
        else:
            raise RuntimeError("reference momentum not implemented")

class TOh3Dp(TOhRep):
    def __init__(self, elements):
        TOhRep.__init__(self, 3)
        self.name = "TOh3D"
        self.mx = gg.gen3D(elements)

if __name__ == "__main__":
    print("for checks execute the test script")

