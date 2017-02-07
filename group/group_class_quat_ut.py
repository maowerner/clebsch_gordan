"""Unit test for the group class
"""

import unittest
import numpy as np

import group_class_quat as gc
import group_class as gcold
import utils
from rotations import mapping

#@unittest.skip("bla")
class TestTOhCMF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.TOh()

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        self.assertEqual(self.group.name, "TO")
        self.assertEqual(self.group.order, 96)
        self.assertEqual(len(self.group.elements), 96)
        self.assertTrue(self.group.faithful)

    # there is no other multiplication table to check against
    #def test_multiplication_table(self):
    #    # map the element numbers to the old numbers
    #    res = np.zeros_like(self.group.tmult_global)
    #    for i in range(res.shape[0]):
    #        for j in range(res.shape[1]):
    #            res[i,j] = mapping.index(self.group.tmult_global[i,j])
    #    # resort the elements
    #    tmapping = [x if x < 48 else x-24 for x in mapping]
    #    tmp = res[tmapping]
    #    tmp = tmp[:,tmapping]
    #    # not equal because elements are sorted different
    #    #self.assertEqual(self.group.tmult, gcold.tcheck0)
    #    self.assertEqual(tmp, gcold.tcheck0)

    def test_inverse_list(self):
        tmp = np.unique(self.group.linv)
        self.assertEqual(len(tmp), self.group.order)

    def test_classes(self):
        self.assertEqual(self.group.nclasses, 16)
        classdims = np.asarray([1,6,8,6,12,1,6,8,6,12,1,8,6,1,8,6])
        self.assertEqual(self.group.cdim, classdims)
        self.assertEqual(self.group.lclasses.shape, (16, 12))

    def test_orthogonality(self):
        res1, res2 = self.group.check_orthogonalities()
        self.assertFalse(res1)
        self.assertFalse(res2)

    def test_su2_characters_1(self):
        res = self.group.characters_of_SU2(1)
        res_theo = np.ones((16,))
        res_theo[5:10] *= -1.
        res_theo[13:] *= -1.
        self.assertEqual(res, res_theo)

    def test_su2_characters_2(self):
        res = self.group.characters_of_SU2(2)
        sq2 = np.sqrt(2)
        res_theo = np.asarray([2.,0.,1.,sq2,0.,-2.,0.,-1.,-sq2,0.,-2.,-1.,-sq2,2.,1.,sq2])
        self.assertEqual(res, res_theo)

#@unittest.skip("bla")
class TestTOhMF1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        tmp = np.asarray([0., 0., 1.])
        self.group = gc.TOh(pref=tmp, debug=0)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_classes(self):
        self.assertEqual(self.group.nclasses, 7)
        classdims = np.asarray([1,2,2,4,4,1,2])
        self.assertEqual(self.group.cdim, classdims)
        self.assertEqual(self.group.lclasses.shape, (7, 4))

    def test_orthogonality(self):
        res1, res2 = self.group.check_orthogonalities()
        self.assertFalse(res1)
        self.assertFalse(res2)

    def test_su2_characters_1(self):
        res = self.group.characters_of_SU2(1)
        res_theo = np.ones((7,))
        res_theo[3:5] *= -1
        self.assertEqual(res, res_theo)

    def test_su2_characters_2(self):
        res = self.group.characters_of_SU2(2)
        sq2 = np.sqrt(2)
        res_theo = np.asarray([2.,0.,sq2,0.,0.,-2.,-sq2])
        self.assertEqual(res, res_theo)

#@unittest.skip("bla")
class TestTOhMF2(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        tmp = np.asarray([1., 1., 0.])
        self.group = gc.TOh(pref=tmp)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_classes(self):
        self.assertEqual(self.group.nclasses, 5)
        classdims = np.asarray([1,2,2,2,1])
        self.assertEqual(self.group.cdim, classdims)
        self.assertEqual(self.group.lclasses.shape, (5, 2))

    def test_orthogonality(self):
        res1, res2 = self.group.check_orthogonalities()
        self.assertFalse(res1)
        self.assertFalse(res2)

    def test_su2_characters_1(self):
        res = self.group.characters_of_SU2(1)
        res_theo = np.ones((5,))
        res_theo[2:4] *= -1.
        self.assertEqual(res, res_theo)

    def test_su2_characters_2(self):
        res = self.group.characters_of_SU2(2)
        res_theo = np.asarray([2.,0.,0.,0.,-2.])
        self.assertEqual(res, res_theo)

#@unittest.skip("bla")
class TestTOhMF3(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        tmp = np.asarray([1., 1., 1.])
        self.group = gc.TOh(pref=tmp)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_classes(self):
        self.assertEqual(self.group.nclasses, 6)
        classdims = np.asarray([1,2,3,3,1,2])
        self.assertEqual(self.group.cdim, classdims)
        self.assertEqual(self.group.lclasses.shape, (6, 3))

    def test_orthogonality(self):
        res1, res2 = self.group.check_orthogonalities()
        self.assertFalse(res1)
        self.assertFalse(res2)

    def test_su2_characters_1(self):
        res = self.group.characters_of_SU2(1)
        res_theo = np.ones((6,))
        res_theo[2] *= -1
        res_theo[3] *= -1
        self.assertEqual(res, res_theo)

    def test_su2_characters_2(self):
        res = self.group.characters_of_SU2(2)
        res_theo = np.asarray([2.,1.,0.,0.,-2.,-1.])
        self.assertEqual(res, res_theo)

#@unittest.skip("bla")
class TestTOh_full_CMF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.TOh(withinversion=True, irreps=True)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_number_irreps(self):
        pass

    def test_1D_representations(self):
        ir = gc.TOh1D(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult, verbose=True))

#@unittest.skip("bla")
class TestTOh_full_MF1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pref = np.asarray([0., 0., 1.])
        self.group = gc.TOh(self.pref, withinversion=True, irreps=True, debug=0)

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
        self.group = gc.TOh(self.pref, withinversion=True, irreps=True)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_1D_representations(self):
        ir = gc.TOh1D(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult))

    def test_working(self):
        self.assertTrue(True)

#@unittest.skip("bla")
class TestTOh_full_MF3(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pref = np.asarray([1., 1., 1.])
        self.group = gc.TOh(self.pref, withinversion=True, irreps=True, debug=0)
        self.group.print_char_table()

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_1D_representations(self):
        ir = gc.TOh1D(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult))

    def test_working(self):
        self.assertTrue(True)

#@unittest.skip("bla")
class TestTOh3Dp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.TOh(withinversion=True)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_bla(self):
        ir = gc.TOh3Dp(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult))
        #print(ir.characters(self.group.crep))

if __name__ == "__main__":
    unittest.main(verbosity=2)
