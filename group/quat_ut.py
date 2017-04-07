"""Unit test for the quaternion class.
"""

import unittest
import numpy as np

import quat
import utils

class TestQNew(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)
        self.vec = np.ones((4,))

    def test_create(self):
        q = quat.QNew()
        self.assertEqual(q.q, np.zeros((4,)))
        self.assertEqual(q.i, 1)

    def test_create_from_vector(self):
        inv = -1
        q = quat.QNew.create_from_vector(self.vec, inv)
        self.assertEqual(q.q, self.vec)
        self.assertEqual(q.i, inv)

    def test_neg(self):
        q1 = quat.QNew.create_from_vector(self.vec, 1)
        q1  = -q1
        self.assertEqual(q1.q, -1*self.vec)
        self.assertEqual(q1.i, -1)

    def test_abs(self):
        q1 = quat.QNew.create_from_vector(self.vec, 1)
        q1abs = abs(q1)
        self.assertEqual(q1abs, 2.)

    def test_mul(self):
        q1 = quat.QNew.create_from_vector(self.vec, 1)
        q2 = quat.QNew.create_from_vector(self.vec, 1)
        q3 = q1*q2
        res_theo = np.asarray([-2., 2., 2., 2.])
        self.assertEqual(q3.q, res_theo)
        self.assertEqual(q3.i, 1)

    def test_conj(self):
        q1 = quat.QNew.create_from_vector(self.vec, 1)
        res_theo = np.asarray([1., -1., -1., -1.])
        q2 = q1.conj()
        self.assertEqual(q2.q, res_theo)
        self.assertEqual(q2.i, 1)

    def test_equal(self):
        q1 = quat.QNew.create_from_vector(self.vec, 1)
        q2 = quat.QNew.create_from_vector(self.vec, 1)

        self.assertTrue(q1 == q2)
        self.assertFalse(q1 != q2)

    def test_not_equal(self):
        q1 = quat.QNew.create_from_vector(self.vec, 1)
        q2 = quat.QNew.create_from_vector(self.vec+1, 1)

        self.assertFalse(q1 == q2)
        self.assertTrue(q1 != q2)

    def test_comp(self):
        q1 = quat.QNew.create_from_vector(self.vec, 1)
        self.assertTrue(q1.comp(self.vec))
        self.assertFalse(q1.comp(self.vec+1.))

    def test_rotation_matrix_unit_normalized(self):
        v = np.asarray([1., 0., 0., 0.])
        q1 = quat.QNew.create_from_vector(v, 1)
        res = q1.rotation_matrix()
        res_theo = np.identity(3)
        self.assertEqual(res, res_theo)

    def test_rotation_matrix_(self):
        q1 = quat.QNew.create_from_vector(self.vec, 1)
        res = q1.rotation_matrix()
        res_theo = np.asarray([
                [0., 0., 1.],
                [1., 0., 0.],
                [0., 1., 0.]])
        self.assertEqual(res, res_theo)

if __name__ == "__main__":
    unittest.main(verbosity=2)
