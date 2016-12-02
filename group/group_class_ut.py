"""Unit test for the group class
"""

import unittest
import numpy as np

import group_class as gc
import utils

class TestOhGroup_CMF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.OhGroup()

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        self.assertEqual(self.group.prec, 1e-6)
        self.assertEqual(self.group.p2, 0)

        self.assertIsNone(self.group.instances)
        self.assertEqual(self.group.irid, -1)
        self.assertIsNone(self.group.dim)
        self.assertIsNone(self.group.mx)
        self.assertFalse(self.group.faithful)

    def test_list_lengths(self):
        # check the length of the lists
        self.assertEqual(len(self.group.lclasses), 8)
        self.assertEqual(len(self.group.lirreps), 8)
        self.assertEqual(len(self.group.lirreps), len(self.group.lclasses))
        self.assertEqual(len(self.group.lrotations), 48)

        # check derived data
        self.assertEqual(self.group.nclasses, 8)
        self.assertEqual(self.group.nirreps, 8)
        self.assertEqual(self.group.nirreps, self.group.nclasses)

        self.assertEqual(self.group.order, 48)

    def test_list_contents(self):
        # check number of elements in list
        tmp = [1, 6, 6, 6, 8, 8, 12, 1]
        self.assertListEqual(self.group.sclass, tmp)

        tmp = [0, 1, 7, 13, 19, 27, 35, 47]
        self.assertListEqual(self.group.rclass, tmp)

    def test_tables(self):
        self.assertEqual(self.group.tclass.size, 48*48)
        self.assertEqual(self.group.tmult.size, 48*48)

        self.assertEqual(self.group.tchar.size, 8*8)
        self.assertEqual(self.group.tcheck1.size, 8*8)
        self.assertEqual(self.group.tcheck2.size, 8*8)

class TestOhGroup_CMF_Instances(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.OhGroup(instances=True)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_instances(self):
        self.assertIsNotNone(self.group.instances)

    def test_char_table(self):
        s2 = np.sqrt(2)
        tmp = [[1, 1,  1,  1, 1, 1,  1, 1],
               [1, 1, -1, -1, 1, 1, -1, 1],
               [3, -1, 1, 1, 0, 0, -1, 3],
               [3, -1, -1, -1, 0, 0, 1, 3],
               [2, 2, 0, 0, -1, -1, 0, 2],
               [2, 0, s2, -s2, 1, -1, 0, -2],
               [2, 0, -s2, s2, 1, -1, 0, -2],
               [4, 0, 0, 0, -1, 1, 0, -4]]
        tmp = np.asarray(tmp, dtype=complex)
        for i in range(tmp.shape[0]):
            irname = self.group.lirreps[i]
            fmessage = "class %s failed:\n%r\n%r" % (irname, tmp[i],
                    self.group.tchar[i])
            self.assertTrue(np.allclose(tmp[i], self.group.tchar[i]),
                    fmessage)

    def test_check1(self):
        tmp = 48*np.identity(8)
        self.assertEqual(tmp, self.group.tcheck1)

    def test_check2(self):
        tmp = [1, 6, 6, 6, 8, 8, 12, 1]
        tmp = np.diag(np.ones((8,))*48/np.asarray(tmp))
        self.assertEqual(tmp, self.group.tcheck2)

    def test_A1(self):
        gA1 = self.group.instances[0]
        self.assertEqual(np.ones((48,1,1), dtype=complex), gA1.mx)
        self.assertEqual(0, gA1.irid)
        self.assertEqual(1, gA1.dim)

    def test_multiplication_table(self):
        res_theo = np.ones((48,),dtype=int)*np.sum(range(48))
        self.assertEqual(np.sum(gc.tcheck0,axis=0),res_theo)
        self.assertEqual(np.sum(gc.tcheck0,axis=1),res_theo)
        self.assertEqual(gc.tcheck0, self.group.tmult)

class TestOhGroup_MF1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.OhGroup(p2=1)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        self.assertEqual(self.group.prec, 1e-6)
        self.assertEqual(self.group.p2, 1)

        self.assertIsNone(self.group.instances)
        self.assertEqual(self.group.irid, -1)
        self.assertIsNone(self.group.dim)
        self.assertIsNone(self.group.mx)
        self.assertFalse(self.group.faithful)

    def test_list_lengths(self):
        # check the length of the lists
        self.assertEqual(len(self.group.lclasses), 7)
        self.assertEqual(len(self.group.lirreps), 7)
        self.assertEqual(len(self.group.lirreps), len(self.group.lclasses))
        self.assertEqual(len(self.group.lrotations), 16)

        # check derived data
        self.assertEqual(self.group.nclasses, 7)
        self.assertEqual(self.group.nirreps, 7)
        self.assertEqual(self.group.nirreps, self.group.nclasses)

        self.assertEqual(self.group.order, 16)

    def test_list_contents(self):
        # check number of elements in list
        tmp = [1, 2, 2, 2, 4, 4, 1]
        self.assertListEqual(self.group.sclass, tmp)

        tmp = [0, 3, 9, 15, 1, 37, 47]
        self.assertListEqual(self.group.rclass, tmp)

    def test_tables(self):
        self.assertEqual(self.group.tclass.size, 16*16)
        self.assertEqual(self.group.tmult.size, 16*16)

        self.assertEqual(self.group.tchar.size, 7*7)
        self.assertEqual(self.group.tcheck1.size, 7*7)
        self.assertEqual(self.group.tcheck2.size, 7*7)

class TestOhGroup_MF1_Instances(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.OhGroup(p2=1, instances=True)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_instances(self):
        self.assertIsNotNone(self.group.instances)

    def test_char_table(self):
        s2 = np.sqrt(2)
        tmp = [[1, 1,  1,  1,  1,  1, 1],
               [1, 1,  1,  1, -1, -1, 1],
               [1, 1, -1, -1,  1, -1, 1],
               [1, 1, -1, -1, -1,  1, 1],
               [2, -2, 0, 0, 0, 0, 2],
               [2, 0, s2, -s2, 0, 0, -2],
               [2, 0, -s2, s2, 0, 0, -2]]
        tmp = np.asarray(tmp, dtype=complex)
        for i in range(tmp.shape[0]):
            irname = self.group.lirreps[i]
            fmessage = "class %s failed:\n%r\n%r" % (irname, tmp[i],
                    self.group.tchar[i])
            self.assertTrue(np.allclose(tmp[i], self.group.tchar[i]),
                    fmessage)

    def test_check1(self):
        tmp = 16*np.identity(7)
        self.assertEqual(tmp, self.group.tcheck1)

    def test_check2(self):
        tmp = [1, 2, 2, 2, 4, 4, 1]
        tmp = np.diag(np.ones((7,))*16/np.asarray(tmp))
        self.assertEqual(tmp, self.group.tcheck2)

    def test_A1(self):
        gA1 = self.group.instances[0]
        self.assertEqual(np.ones((16,1,1), dtype=complex), gA1.mx)
        self.assertEqual(0, gA1.irid)
        self.assertEqual(1, gA1.dim)

    def test_multiplication_table(self):
        res_theo = np.ones((16,),dtype=int)*np.sum(range(16))
        self.assertEqual(np.sum(gc.tcheck1,axis=0),res_theo)
        self.assertEqual(np.sum(gc.tcheck1,axis=1),res_theo)
        self.assertEqual(gc.tcheck1, self.group.tmult)

class TestOhGroup_MF2(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.OhGroup(p2=2)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        self.assertEqual(self.group.prec, 1e-6)
        self.assertEqual(self.group.p2, 2)

        self.assertIsNone(self.group.instances)
        self.assertEqual(self.group.irid, -1)
        self.assertIsNone(self.group.dim)
        self.assertIsNone(self.group.mx)
        self.assertFalse(self.group.faithful)

    def test_list_lengths(self):
        # check the length of the lists
        self.assertEqual(len(self.group.lclasses), 5)
        self.assertEqual(len(self.group.lirreps), 5)
        self.assertEqual(len(self.group.lirreps), len(self.group.lclasses))
        self.assertEqual(len(self.group.lrotations), 8)

        # check derived data
        self.assertEqual(self.group.nclasses, 5)
        self.assertEqual(self.group.nirreps, 5)
        self.assertEqual(self.group.nirreps, self.group.nclasses)

        self.assertEqual(self.group.order, 8)

    def test_list_contents(self):
        # check number of elements in list
        tmp = [1, 2, 2, 2, 1]
        self.assertListEqual(self.group.sclass, tmp)

        tmp = [0, 37, 3, 38, 47]
        self.assertListEqual(self.group.rclass, tmp)

    def test_tables(self):
        self.assertEqual(self.group.tclass.size, 8*8)
        self.assertEqual(self.group.tmult.size, 8*8)

        self.assertEqual(self.group.tchar.size, 5*5)
        self.assertEqual(self.group.tcheck1.size, 5*5)
        self.assertEqual(self.group.tcheck2.size, 5*5)

class TestOhGroup_MF2_Instances(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.OhGroup(p2=2, instances=True)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_instances(self):
        self.assertIsNotNone(self.group.instances)

    def test_char_table(self):
        s2 = np.sqrt(2)
        tmp = [[1, 1,  1,  1, 1],
               [1, 1, -1, -1, 1],
               [1, -1, -1, 1, 1],
               [1, -1, 1, -1, 1],
               [2, 0, 0, 0, -2]]
        tmp = np.asarray(tmp, dtype=complex)
        for i in range(tmp.shape[0]):
            irname = self.group.lirreps[i]
            fmessage = "class %s failed:\n%r\n%r" % (irname, tmp[i],
                    self.group.tchar[i])
            self.assertTrue(np.allclose(tmp[i], self.group.tchar[i]),
                    fmessage)

    def test_check1(self):
        tmp = 8*np.identity(5)
        self.assertEqual(tmp, self.group.tcheck1)

    def test_check2(self):
        tmp = [1, 2, 2, 2, 1]
        tmp = np.diag(np.ones((5,))*8/np.asarray(tmp))
        self.assertEqual(tmp, self.group.tcheck2)

    def test_A1(self):
        gA1 = self.group.instances[0]
        self.assertEqual(np.ones((8,1,1), dtype=complex), gA1.mx)
        self.assertEqual(0, gA1.irid)
        self.assertEqual(1, gA1.dim)

    def test_multiplication_table(self):
        res_theo = np.ones((8,),dtype=int)*np.sum(range(8))
        self.assertEqual(np.sum(gc.tcheck2,axis=0),res_theo)
        self.assertEqual(np.sum(gc.tcheck2,axis=1),res_theo)
        self.assertEqual(gc.tcheck2, self.group.tmult)

class TestOhGroup_MF3(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.OhGroup(p2=3)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_attributes(self):
        self.assertEqual(self.group.prec, 1e-6)
        self.assertEqual(self.group.p2, 3)

        self.assertIsNone(self.group.instances)
        self.assertEqual(self.group.irid, -1)
        self.assertIsNone(self.group.dim)
        self.assertIsNone(self.group.mx)
        self.assertFalse(self.group.faithful)

    def test_list_lengths(self):
        # check the length of the lists
        self.assertEqual(len(self.group.lclasses), 6)
        self.assertEqual(len(self.group.lirreps), 6)
        self.assertEqual(len(self.group.lirreps), len(self.group.lclasses))
        self.assertEqual(len(self.group.lrotations), 12)

        # check derived data
        self.assertEqual(self.group.nclasses, 6)
        self.assertEqual(self.group.nirreps, 6)
        self.assertEqual(self.group.nirreps, self.group.nclasses)

        self.assertEqual(self.group.order, 12)

    def test_list_contents(self):
        # check number of elements in list
        tmp = [1, 2, 2, 3, 3, 1]
        self.assertListEqual(self.group.sclass, tmp)

        tmp = [0, 19, 27, 36, 38, 47]
        self.assertListEqual(self.group.rclass, tmp)

    def test_tables(self):
        self.assertEqual(self.group.tclass.size, 12*12)
        self.assertEqual(self.group.tmult.size, 12*12)

        self.assertEqual(self.group.tchar.size, 6*6)
        self.assertEqual(self.group.tcheck1.size, 6*6)
        self.assertEqual(self.group.tcheck2.size, 6*6)

class TestOhGroup_MF3_Instances(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.group = gc.OhGroup(p2=3, instances=True)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, utils.check_array)

    def test_instances(self):
        self.assertIsNotNone(self.group.instances)

    def test_char_table(self):
        s2 = np.sqrt(2)
        tmp = [[1, 1,  1,  1,  1, 1],
               [1, 1,  1, -1, -1, 1],
               [1, -1, 1,  1j, -1j, -1],
               [1, -1, 1, -1j,  1j, -1],
               [2, -1, -1, 0, 0, 2],
               [2, 1, -1, 0, 0, -2]]
        tmp = np.asarray(tmp, dtype=complex)
        for i in range(tmp.shape[0]):
            irname = self.group.lirreps[i]
            fmessage = "class %s failed:\n%r\n%r" % (irname, tmp[i],
                    self.group.tchar[i])
            self.assertTrue(np.allclose(tmp[i], self.group.tchar[i]),
                    fmessage)

    def test_check1(self):
        tmp = 12*np.identity(6)
        self.assertEqual(tmp, self.group.tcheck1)

    def test_check2(self):
        tmp = [1, 2, 2, 3, 3, 1]
        tmp = np.diag(np.ones((6,))*12/np.asarray(tmp))
        self.assertEqual(tmp, self.group.tcheck2)

    def test_A1(self):
        gA1 = self.group.instances[0]
        self.assertEqual(np.ones((12,1,1), dtype=complex), gA1.mx)
        self.assertEqual(0, gA1.irid)
        self.assertEqual(1, gA1.dim)

    def test_multiplication_table(self):
        res_theo = np.ones((12,),dtype=int)*np.sum(range(12))
        self.assertEqual(np.sum(gc.tcheck3,axis=0),res_theo)
        self.assertEqual(np.sum(gc.tcheck3,axis=1),res_theo)
        self.assertEqual(gc.tcheck3, self.group.tmult)

if __name__ == "__main__":
    unittest.main(verbosity=2)
