"""Class for the basis of a group."""

import numpy as np

import utils
from rotations import _all_rotations

class BasisIrrep(object):
    def __init__(self, j=None, group=None, prec=1e-6, reading=False):
        if reading:
            return
        if group.instances is None:
            raise RuntimeError("the group has no irrep instances")
        self.p2 = group.p2
        self.prec = prec
        self.jmax = j
        self.verbose = 0
        self.lm = [(l,m) for l in range(j) for m in range(-l,l+1)]
        self.IR = [(group.lirreps[i.irid], row) for i in group.instances for row in range(i.dim)]
        self.irreps = group.lirreps
        self.dims = (1, len(self.IR), len(self.lm))
        # basis[multi][irrep/row][l/m]
        self.basis = np.zeros(self.dims, dtype=complex)
        for j in range(self.jmax):
            if self.verbose > 0:
                print("getting basis for j=%.1f" % j)
            for m, ir in self.subduce(group, j):
                if self.verbose > 0:
                    print("working on irrep %s" % (ir.lirreps[ir.irid]))
                # check if multiplicities are enough
                if m > self.dims[0]:
                    tmp = list(self.dims)
                    tmp[0] = m-self.dims[0]
                    self.basis = np.vstack((self.basis, np.zeros(tmp)))
                    self.dims = self.basis.shape
                self._get_basis(ir, m, j)
        #print(self.basis.shape)

    @classmethod
    def read(cls, fname):
        # get data from file
        fh = np.load(fname)
        tmp = fh['param']
        basis = fh['basis']
        del fh
        # create class
        c = cls(reading=True)
        c.lm = tmp[1]
        c.IR = tmp[2]
        c.irreps = tmp[3]
        c.prec = tmp[0][0]
        c.jmax = tmp[0][1]
        c.p2 = tmp[0][2]
        c.verbose = tmp[0][3]
        c.basis = basis
        c.dims = basis.shape
        return c

    def save(self, fname):
        tmp = []
        tmp.append((self.prec, self.jmax, self.p2, self.verbose))
        tmp.append(self.lm)
        tmp.append(self.IR)
        tmp.append(self.irreps)
        tmp = np.asarray(tmp, dtype=object)
        np.savez(fname, basis=self.basis, param=tmp)

    def get_ir_index(self, irrep, row):
        ind = None
        try:
            ind = self.IR.index((irrep, row))
        except:
            pass
        return ind

    def get_ir_index_rev(self, irrep):
        ind = None
        try:
            ind = self.IR.index((irrep, 0))
            while ind+1 < len(self.IR) and self.IR[ind+1][0] == irrep:
                ind += 1
        except:
            pass
        return ind

    def get_lm_index(self, l, m):
        ind = None
        try:
            ind = self.lm.index((l, m))
        except:
            pass
        return ind

    def display(self):
        if self.basis.shape[0] != 1:
            multi = True
        else:
            multi=False
        def _s(x):
            try:
                tmp = sympy.nsimplify(x)
                tmp1 = sympy.simplify(tmp)
                # TODO: does no align properly using rjust
                # maybe due to implicit string conversion
            except ImportError:
                tmp1 = x
            return str(tmp1)
        for j in range(self.jmax):
            print("")
            print(" basis for j %d ".center(30, "*") % j)
            header = "".join(["%d".rjust(10) % m for m in range(-j,j+1)])
            if not multi:
                header = "".join(("\nIR R ", header))
            else:
                header = "".join(("\nIR R M ", header))
            print(header)
            # get indices for spin
            idlml = self.get_lm_index(j, -j)
            idlmh = self.get_lm_index(j, j)
            for ir in self.irreps:
                # get indices for irrep
                idirl = self.get_ir_index(ir, 0)
                idirh = self.get_ir_index_rev(ir)
                # loop over multiplicities
                for m, mb in enumerate(self.basis[:,idirl:idirh+1,idlml:idlmh+1]):
                    # check if all entries are zero and continue if so
                    if not np.any(np.abs(mb) > self.prec):
                        continue
                    for r, b in enumerate(mb):
                        if multi:
                            prefix = "%2s %d %d" % (ir, r, m)
                        else:
                            prefix = "%2s %d" % (ir, r)
                        text = "".join(["%s".rjust(10) % (_s(x)) for x in b])
                        text = "".join((prefix, text))
                        print(text)

    def coefficient(self, irrep, row, l, mult=0):
        indir = self.get_ir_index(irrep, row)
        indlml = self.get_lm_index(l, -l)
        indlmh = self.get_lm_index(l, l)
        if indir is None or indlml is None or indlmh is None:
            res = None
        else:
            res = self.basis[mult, indir, indlml:indlmh+1]
        return res

    def _bvecs(self, irrep, row, j):
        pre = float(irrep.dim)/irrep.order
        tmp = np.zeros((int(2*j+1), int(2*j+1)), dtype=complex)
        for k, index in enumerate(irrep.lrotations):
            coeff = irrep.mx[k][row, row].conj()
            u = _all_rotations[index].u_matrix(j)
            tmp += u*coeff
        return (pre*tmp)

    def _get_basis(self, irrep, multiplicity, j):
        indlml = self.get_lm_index(j, -j)
        indlmh = self.get_lm_index(j, j)
        for row in range(irrep.dim):
            if self.verbose > 0:
                print(" starting row %d ".center(40, "_") % row)
            m = j
            mult = 0
            indir = self.get_ir_index(irrep.lirreps[irrep.irid], row)
            # get all possible base vectors and check for small entries
            b = self._bvecs(irrep, row, j)
            b = utils.clean_complex(b, self.prec)
            while m > -j-1 and mult < multiplicity:
                if self.verbose > 0:
                    print("  using m=%d, searching for multiplicity %d" % (m, mult))
                _b = b[:,j+m]
                # set small values explicitly to 0
                # check if all entries are zero
                norm = np.sqrt(np.vdot(_b, _b))
                if norm < self.prec:
                    if self.verbose > 0:
                        print("  norm is zero, continuing")
                    m -= 1
                    continue
                _b /= norm

                # check against already calculated basis
                accept = self._check_against_old(_b, (indlml, indlmh))
                if accept == True:
                    if self.verbose > 0:
                        print("  accepted basis")
                    self.basis[mult][indir][indlml:indlmh+1] = _b
                    mult += 1
                else:
                    if self.verbose > 0:
                        print("  did not accept basis")
                m -= 1

    def _check_against_old(self, b, ind):
        # iterate over multiplicities
        for i1, b1 in enumerate(self.basis[:,:,ind[0]:ind[1]+1]):
            # loop over irreps
            for i2, b2 in enumerate(b1):
                # check if basis is non-zero
                if np.vdot(b2, b2) > self.prec:
                    check = self._check_ortho(b2, b)
                    if self.verbose > 1:
                        print("    check is %r" % check)
                    if check is False:
                        return check
        return True

    def _check_ortho(self, b1, b2):
        check = np.vdot(b1, b2)
        if np.abs(check) < self.prec:
            return True
        else:
            return False

    def subduce(self, group, j):
        # get the rotation angles of the conjugacy classes
        omegas = [_all_rotations[i].omega for i in group.rclass]
        omegas = np.asarray(omegas)

        # subduce spin j
        characters = np.zeros_like(omegas, dtype=complex)
        for k in range(int(2*j+1)):
            characters += np.exp(1.j*(j-k)*omegas)
        characters = utils.clean_complex(characters, self.prec)

        # calculate multiplicities
        multi = group.tchar.dot(characters*group.sclass)
        multi = np.real_if_close(np.rint(multi/float(group.order)))

        # check and yield the irrep
        check = multi.dot(group.tchar)
        if np.any((check-characters) > self.prec):
            print("characters:")
            print(characters)
            print("multiplicities:")
            print(multi)
            print("check = %r" % check)
            raise RuntimeError("subduction does not work!")
        for m, ir in zip(multi, group.instances):
            if m > self.prec:
                yield (m, ir)

if __name__ == "__main__":
    print("for checks execute the test script")
