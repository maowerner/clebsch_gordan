"""Unit test for the group class
"""

import unittest
import numpy as np

import group_class_quat as gc
import group_class as gcold
import utils

#@unittest.skip("bla")
class TestTOh(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.TOh()

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

    def test_su2_characters_j1(self):
        res = self.group.characters_of_SU2(1, self.group.crep[0])
        res_theo = np.ones((8,))
        self.assertEqual(res, res_theo)

    def test_su2_characters_j2(self):
        res = self.group.characters_of_SU2(2, self.group.crep[0])
        sq2 = np.sqrt(2)
        res_theo = np.asarray([2.,0.,1.,sq2,0.,-2.,-1.,-sq2])
        self.assertEqual(res, res_theo)

#@unittest.skip("bla")
class TestTOh_full(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.TOh(withinversion=True, irreps=True, debug=2)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_1D_representations(self):
        ir = gc.TOh1D(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult, verbose=True))

    def test_working(self):
        self.assertTrue(True)

#@unittest.skip("bla")
class TestTOh_full_MF1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pref = np.asarray([0., 0., 1.])
        self.group = gc.TOh(self.pref, withinversion=True, irreps=True, debug=2)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_1D_representations(self):
        ir = gc.TOh1D(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult, verbose=True))

    def test_working(self):
        self.assertTrue(True)

#@unittest.skip("bla")
class TestTOh_full_MF2(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pref = np.asarray([1., 1., 0.])
        self.group = gc.TOh(self.pref, withinversion=True, irreps=True, debug=2)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_1D_representations(self):
        ir = gc.TOh1D(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult, verbose=True))

    def test_working(self):
        self.assertTrue(True)

class TestTOh_full_MF3(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pref = np.asarray([1., 1., 1.])
        self.group = gc.TOh(self.pref, withinversion=True, irreps=True, debug=2)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_1D_representations(self):
        ir = gc.TOh1D(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult, verbose=True))

    def test_working(self):
        self.assertTrue(True)

#@unittest.skip("bla")
class TestTOhMF1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        tmp = np.asarray([0., 0., 1.])
        self.wi = False
        debug = 1
        self.group = gc.TOh(pref=tmp, withinversion=self.wi)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_nclasses(self):
        res_theo = 14 if self.wi else 7
        self.assertEqual(len(self.group.lclasses), res_theo)

if __name__ == "__main__":
    unittest.main(verbosity=2)
