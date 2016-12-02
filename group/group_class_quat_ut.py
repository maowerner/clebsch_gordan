"""Unit test for the group class
"""

import unittest
import numpy as np

import group_class_quat as gc
import group_class as gcold
import utils

class TestTO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.TO()

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        self.assertEqual(self.group.name, "TO")
        self.assertEqual(self.group.order, 48)
        self.assertEqual(len(self.group.elements), 48)
        self.assertTrue(self.group.faithful)

    #def test_multiplication_table(self):
    #    # not equal because elements are sorted different
    #    self.assertEqual(self.group.tmult, gcold.tcheck0)

    #def test_print(self):
    #    self.group.print_mult_table()

    def test_inverse_list(self):
        tmp = np.unique(self.group.linv)
        self.assertEqual(len(tmp), self.group.order)

    def test_number_classes(self):
        self.assertEqual(self.group.nclasses, 8)
        #print(self.group.cdim)
        #print(self.group.crep)
        #print(self.group.lclasses)
        #self.group.print_class_members()

    def test_orthogonality(self):
        res1, res2 = self.group.check_orthogonalities()
        self.assertFalse(res1)
        self.assertFalse(res2)

    def test_su2_characters_j0(self):
        res = self.group.characters_of_SU2(0, self.group.crep[0])
        res_theo = np.zeros((8,))
        self.assertEqual(res, res_theo)

    def test_su2_characters_j2(self):
        res = self.group.characters_of_SU2(2, self.group.crep[0])
        sq2 = np.sqrt(2)
        res_theo = np.asarray([2.,0.,1.,sq2,0.,-2.,-1.,-sq2])
        self.assertEqual(res, res_theo)

if __name__ == "__main__":
    unittest.main(verbosity=2)
