"""Class for the quaternions with inversion."""

import numpy as np

# quaternion parameters for the group O from Table 71.1 in:
# Simon L. Altmann, Peter Herzig, "Point-Group Theory Tables", 
# Second Edition (corrected), Wien (2011)  
V12 = np.sqrt(0.5) # sqrt(1/2)
# [[ lambda, Lambda_1, Lambda_2, Lambda_3 ]]
qPar = [[ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 0.0, 0.0, 1.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ],
        [ 0.5, 0.5, 0.5, 0.5 ],
        [ 0.5,-0.5,-0.5, 0.5 ],
        [ 0.5, 0.5,-0.5,-0.5 ],
        [ 0.5,-0.5, 0.5,-0.5 ],
        [ 0.5,-0.5,-0.5,-0.5 ],
        [ 0.5, 0.5, 0.5,-0.5 ],
        [ 0.5,-0.5, 0.5, 0.5 ],
        [ 0.5, 0.5,-0.5, 0.5 ],
        [ V12, V12, 0.0, 0.0 ],
        [ V12, 0.0, V12, 0.0 ],
        [ V12, 0.0, 0.0, V12 ],
        [ V12,-V12, 0.0, 0.0 ],
        [ V12, 0.0,-V12, 0.0 ],
        [ V12, 0.0, 0.0,-V12 ],
        [ 0.0, V12, V12, 0.0 ],
        [ 0.0,-V12, V12, 0.0 ],
        [ 0.0, V12, 0.0, V12 ],
        [ 0.0, 0.0,-V12,-V12 ],
        [ 0.0, V12, 0.0,-V12 ],
        [ 0.0, 0.0,-V12, V12 ]]

class QNew(object):
    def __init__(self):
        self.q = np.zeros((4,))
        self.i = int(1)

    @classmethod
    def create_from_vector(cls, vector, inversion):
        tmp = cls()
        _vec = np.asarray(vector)
        tmp.q = _vec.copy()
        _inv = int(inversion)
        tmp.i = _inv
        return tmp

    def __add__(self, other):
        if isinstance(other, QNew):
            tmpvec = self.q + other.q
            tmpinv = self.i * other.i
            return QNew.create_from_vector(tmpvec, tmpinv)
        else:
            raise NotImplementedError

    def __iadd__(self, other):
        if isinstance(other, QNew):
            self.q += other.q
            self.i *= other.i
            return self
        else:
            raise NotImplementedError

    def __abs__(self):
        return np.sqrt(np.dot(self.q, self.q))

    def __neg__(self):
        self.q = -self.q
        self.i = -self.i
        return self

    def __mul__(self, other):
        q1 = self.q
        q2 = other.q
        tvec = np.zeros_like(q1)
        tvec[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        tvec[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        tvec[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        tvec[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        tinv = self.i * other.i
        return QNew.create_from_vector(tvec, tinv)

    def conj(self):
        tvec = self.q * np.asarray([1., -1., -1., -1.])
        return QNew.create_from_vector(tvec, self.i)

    def norm(self):
        return np.dot(self.q, self.q)

    # code inspired by the quaternion package of moble
    # https://github.com/moble/quaternion
    def rotation_matrix(self):
        n = self.norm()
        if np.abs(n) < self.prec:
            raise ZeroDivisionError("Norm of quaternion is zero.")
        _q = self.q
        if np.abs(1-n) < self.prec:
            res = np.array(
                [[1-2*(_q[2]**2 + _q[3]**2), 2*(_q[1]*_q[2] - _q[3]*_q[0]),
                    2*(_q[1]*_q[3] + _q[2]*_q[0])],
                 [2*(_q[1]*_q[2] + _q[3]*_q[0]), 1-2*(_q[1]**2 + _q[3]**2),
                    2*(_q[2]*_q[3] - _q[1]*_q[0])],
                 [2*(_q[1]*_q[3] - _q[2]*_q[0]), 2*(_q[2]*_q[3] + _q[1]*_q[0]),
                    1-2*(_q[1]**2 + _q[2]**2)]])
        else:
            res = np.array(
                [[1-2*(_q[2]**2 + _q[3]**2)/n, 2*(_q[1]*_q[2] - _q[3]*_q[0])/n,
                    2*(_q[1]*_q[3] + _q[2]*_q[0])/n],
                 [2*(_q[1]*_q[2] + _q[3]*_q[0])/n, 1-2*(_q[1]**2 + _q[3]**2)/n,
                    2*(_q[2]*_q[3] - _q[1]*_q[0])/n],
                 [2*(_q[1]*_q[3] - _q[2]*_q[0])/n, 2*(_q[2]*_q[3] + _q[1]*_q[0])/n,
                    1-2*(_q[1]**2 + _q[2]**2)/n]])
        return res

if __name__ == "__main__":
    print("for checks execute the test script")
