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
        self.assertEqual(self.group.nclasses, len(self.group.irreps))

    def test_1D_representations(self):
        ir = gc.TOh1D(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult, verbose=True))

    def test_restore_irreps(self):
        tmp_tchar = self.group.tchar.copy()
        tmp_irdim = self.group.irrepdim.copy()
        tmp_suff = list(self.group.suffixes)
        tmp_irnames = list(self.group.irrepsname)
        self.group.tchar.fill(0.)
        self.group.restore_irreps()
        self.assertEqual(self.group.tchar, tmp_tchar)
        self.assertEqual(self.group.irrepdim, tmp_irdim)
        self.assertEqual(self.group.suffixes, tmp_suff)
        self.assertEqual(self.group.irrepsname, tmp_irnames)

    def test_save_and_read(self):
        self.group.save()
        g = gc.TOh.read(p2=0)
        self.assertEqual(self.group.tchar, g.tchar)
        self.assertEqual(self.group.irrepdim, g.irrepdim)
        self.assertEqual(self.group.suffixes, g.suffixes)
        self.assertEqual(self.group.irrepsname, g.irrepsname)
        self.assertEqual(self.group.tmult, g.tmult)
        self.assertEqual(self.group.tmult_global, g.tmult_global)
        self.assertEqual(self.group.flip, g.flip)
        self.assertEqual(self.group.crep, g.crep)

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

    def test_number_irreps(self):
        self.assertEqual(self.group.nclasses, len(self.group.irreps))

    def test_restore_irreps(self):
        tmp_tchar = self.group.tchar.copy()
        tmp_irdim = self.group.irrepdim.copy()
        tmp_suff = list(self.group.suffixes)
        tmp_irnames = list(self.group.irrepsname)
        self.group.tchar.fill(0.)
        self.group.restore_irreps()
        self.assertEqual(self.group.tchar, tmp_tchar)
        self.assertEqual(self.group.irrepdim, tmp_irdim)
        self.assertEqual(self.group.suffixes, tmp_suff)
        self.assertEqual(self.group.irrepsname, tmp_irnames)

    def test_save_and_read(self):
        self.group.save()
        g = gc.TOh.read(p2=1)
        self.assertEqual(self.group.tchar, g.tchar)
        self.assertEqual(self.group.irrepdim, g.irrepdim)
        self.assertEqual(self.group.suffixes, g.suffixes)
        self.assertEqual(self.group.irrepsname, g.irrepsname)
        self.assertEqual(self.group.tmult, g.tmult)
        self.assertEqual(self.group.tmult_global, g.tmult_global)
        self.assertEqual(self.group.flip, g.flip)
        self.assertEqual(self.group.crep, g.crep)

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

    def test_number_irreps(self):
        self.assertEqual(self.group.nclasses, len(self.group.irreps))

    def test_restore_irreps(self):
        tmp_tchar = self.group.tchar.copy()
        tmp_irdim = self.group.irrepdim.copy()
        tmp_suff = list(self.group.suffixes)
        tmp_irnames = list(self.group.irrepsname)
        self.group.tchar.fill(0.)
        self.group.restore_irreps()
        self.assertEqual(self.group.tchar, tmp_tchar)
        self.assertEqual(self.group.irrepdim, tmp_irdim)
        self.assertEqual(self.group.suffixes, tmp_suff)
        self.assertEqual(self.group.irrepsname, tmp_irnames)

    def test_save_and_read(self):
        self.group.save()
        g = gc.TOh.read(p2=2)
        self.assertEqual(self.group.tchar, g.tchar)
        self.assertEqual(self.group.irrepdim, g.irrepdim)
        self.assertEqual(self.group.suffixes, g.suffixes)
        self.assertEqual(self.group.irrepsname, g.irrepsname)
        self.assertEqual(self.group.tmult, g.tmult)
        self.assertEqual(self.group.tmult_global, g.tmult_global)
        self.assertEqual(self.group.flip, g.flip)
        self.assertEqual(self.group.crep, g.crep)

#@unittest.skip("bla")
class TestTOh_full_MF3(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pref = np.asarray([1., 1., 1.])
        self.group = gc.TOh(self.pref, withinversion=True, irreps=True, debug=0)
        #self.group.print_char_table()

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_1D_representations(self):
        ir = gc.TOh1D(self.group.elements)
        self.assertTrue(ir.is_representation(self.group.tmult))

    def test_number_irreps(self):
        self.assertEqual(self.group.nclasses, len(self.group.irreps))

    def test_restore_irreps(self):
        tmp_tchar = self.group.tchar.copy()
        tmp_irdim = self.group.irrepdim.copy()
        tmp_suff = list(self.group.suffixes)
        tmp_suff_i = list(self.group.suffixes_i)
        tmp_irnames = list(self.group.irrepsname)
        self.group.tchar.fill(0.)
        self.group.restore_irreps()
        self.assertEqual(self.group.tchar, tmp_tchar)
        self.assertEqual(self.group.irrepdim, tmp_irdim)
        self.assertEqual(self.group.suffixes, tmp_suff)
        self.assertEqual(self.group.suffixes_i, tmp_suff_i)
        self.assertEqual(self.group.irrepsname, tmp_irnames)

    def test_save_and_read(self):
        self.group.save()
        g = gc.TOh.read(p2=3)
        self.assertEqual(self.group.tchar, g.tchar)
        self.assertEqual(self.group.irrepdim, g.irrepdim)
        self.assertEqual(self.group.suffixes, g.suffixes)
        self.assertEqual(self.group.irrepsname, g.irrepsname)
        self.assertEqual(self.group.tmult, g.tmult)
        self.assertEqual(self.group.tmult_global, g.tmult_global)
        self.assertEqual(self.group.flip, g.flip)
        self.assertEqual(self.group.crep, g.crep)

if __name__ == "__main__":
    unittest.main(verbosity=2)
