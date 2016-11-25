"""Unit test for rotations
"""

import unittest
import numpy as np

import utils
import rotations as rot

_prec=1e-10

class TestRotObj_Functions(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)
        # vector, norm, rotation angle
        self.v = np.asarray([1., 1., 0.])
        self.n = np.sqrt(2.)
        self.o = np.pi
        # theta, phi
        self.t = np.pi/2. #np.arccos(0.)
        self.p = np.pi/4. #np.arctan2(1., 1.)
        # rotation object
        self.r = rot.RotObj(self.v, self.o)

    def test_get_functions(self):
        tmp = self.r.get_vector()
        self.assertEqual(tmp, self.v/self.n)
        self.assertEqual(self.o, self.r.get_omega())
        self.assertTupleEqual((self.t, self.p), self.r.get_angles())

    def test_attributes(self):
        tmpv = complex(1.)
        tmpu = complex(0.)
        self.assertAlmostEqual(self.r.u, tmpu, delta=_prec)
        self.assertAlmostEqual(self.r.v, tmpv, delta=_prec)

    def test_u_element_000(self):
        res_theo = complex(1.)
        res = self.r.u_element(0, 0, 0)
        self.assertAlmostEqual(res_theo, res, delta=_prec)

    def test_u_element_100(self):
        res_theo = complex(-1.)
        res = self.r.u_element(1, 0, 0)
        self.assertAlmostEqual(res_theo, res, delta=_prec)

    def test_u_element_111(self):
        res_theo = complex(0.)
        res = self.r.u_element(1, 1, 1)
        self.assertAlmostEqual(res_theo, res, delta=_prec)

    def test_u_element_110(self):
        res_theo = complex(0.)
        res = self.r.u_element(1, 1, 0)
        self.assertAlmostEqual(res_theo, res, delta=_prec)

    def test_u_matrix_0(self):
        res_theo = np.ones((1,1), dtype=complex)
        res = self.r.u_matrix(0)
        self.assertEqual(res_theo, res)

    def test_u_matrix_1(self):
        res_theo = np.zeros((3,3), dtype=complex)
        res_theo[1,1]= -1.
        res_theo[0,2]= -1.j
        res_theo[2,0]= +1.j
        res = self.r.u_matrix(1)
        self.assertEqual(res_theo, res)

    def test_rotate_vector_parallel(self):
        res = self.r.rot_vector(self.v)
        self.assertEqual(res, self.v)

    def test_rotate_vector_perpendicular(self):
        vec = np.asarray([1., -1., 0])
        res_theo = -vec
        res = self.r.rot_vector(vec)
        self.assertEqual(res, res_theo)

    def test_rotate_vector_100(self):
        vec = np.asarray([1., 0., 0])
        res_theo = np.asarray([0., 1., 0])
        res = self.r.rot_vector(vec)
        self.assertEqual(res, res_theo)

    def test_rotate_vector_00m1(self):
        vec = np.asarray([0., 0., -1.])
        r = rot.RotObj(-vec, np.pi)
        res = r.rot_vector(vec)
        self.assertEqual(res, vec)

if __name__ == "__main__":
    unittest.main(verbosity=2)
