"""Unit test for rotations
"""

import unittest
import numpy as np

import utils
import rotations as rot
import quat

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

class TestAllRotations_100(unittest.TestCase):
    def setUp(self):
        self.v = np.asarray([1.,0.,0.])
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_rotations_to_self(self):
        elements = [0, 1, 4, 7, 10, 13, 16, 47]
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r" % (el, self.v, res)
            self.assertEqual(self.v, res, msg=msg)

    def test_rotations_to_neg_self(self):
        elements = [2, 3, 5, 6, 35, 36, 41, 42]
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r" % (el, self.v, res)
            self.assertEqual(-self.v, res, msg="element %d failed" % el)

    def test_rotations_to_y(self):
        elements = [12, 15, 23, 25, 27, 29, 37, 43]
        res_theo = np.zeros((3,))
        res_theo[1] = 1.
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r\nexpected:\n%r" % (
                    el, self.v, res, res_theo)
            self.assertEqual(res_theo, res, msg=msg)

    def test_rotations_to_neg_y(self):
        elements = [9, 18, 20, 22, 32, 34, 38, 44]
        res_theo = np.zeros((3,))
        res_theo[1] = -1.
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r\nexpected:\n%r" % (
                    el, self.v, res, res_theo)
            self.assertEqual(res_theo, res, msg=msg)

    def test_rotations_to_z(self):
        elements = [8, 17, 19, 26, 30, 31, 39, 45]
        res_theo = np.zeros((3,))
        res_theo[2] = 1.
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r\nexpected:\n%r" % (
                    el, self.v, res, res_theo)
            self.assertEqual(res_theo, res, msg=msg)

    def test_rotations_to_neg_z(self):
        elements = [11, 14, 21, 24, 28, 33, 40, 46]
        res_theo = np.zeros((3,))
        res_theo[2] = -1.
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r\nexpected:\n%r" % (
                    el, self.v, res, res_theo)
            self.assertEqual(res_theo, res, msg=msg)

class TestAllRotations_001(unittest.TestCase):
    def setUp(self):
        self.v = np.asarray([0.,0.,1.])
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_rotations_to_self(self):
        elements = [0, 3, 6, 9, 12, 15, 18, 47]
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r" % (el, self.v, res)
            self.assertEqual(self.v, res, msg=msg)

    def test_rotations_to_neg_self(self):
        elements = [1, 2, 4, 5, 37, 38, 43, 44]
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r" % (el, self.v, res)
            self.assertEqual(-self.v, res, msg="element %d failed" % el)

    def test_rotations_to_y(self):
        elements = [7, 16, 19, 24, 28, 31, 35, 41]
        res_theo = np.zeros((3,))
        res_theo[1] = 1.
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r\nexpected:\n%r" % (
                    el, self.v, res, res_theo)
            self.assertEqual(res_theo, res, msg=msg)

    def test_rotations_to_neg_y(self):
        elements = [10, 13, 21, 26, 30, 33, 36, 42]
        res_theo = np.zeros((3,))
        res_theo[1] = -1.
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r\nexpected:\n%r" % (
                    el, self.v, res, res_theo)
            self.assertEqual(res_theo, res, msg=msg)

    def test_rotations_to_x(self):
        elements = [11, 14, 22, 23, 27, 34, 39, 45]
        res_theo = np.zeros((3,))
        res_theo[0] = 1.
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r\nexpected:\n%r" % (
                    el, self.v, res, res_theo)
            self.assertEqual(res_theo, res, msg=msg)

    def test_rotations_to_neg_x(self):
        elements = [8, 17, 20, 25, 29, 32, 40, 46]
        res_theo = np.zeros((3,))
        res_theo[0] = -1.
        for el in elements:
            res = rot._all_rotations[el].rot_vector(self.v)
            msg = "element %d failed:\n%r\nmapped to\n%r\nexpected:\n%r" % (
                    el, self.v, res, res_theo)
            self.assertEqual(res_theo, res, msg=msg)

    def test_mapping_to_quat(self):
        elements = [x for x in range(48)]
        mapped = []
        for el in elements:
            q = rot._all_rotations[el].to_quaternion()
            for i, v in enumerate(quat.qPar):
                if q.comp(v):
                    mapped.append(i)
                elif q.comp(np.negative(v)):
                    mapped.append(i+48)
        print(mapped)
        self.assertEqual(len(mapped), 48)

if __name__ == "__main__":
    unittest.main(verbosity=2)
