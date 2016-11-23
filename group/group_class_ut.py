"""Unit test for the group class
"""

import unittest
import numpy as np

import group_class as gc

class TestOhGroup_CMF(unittest.TestCase):
    def setUp(self):
        self.group = gc.OhGroup()

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
    def setUp(self):
        self.group = gc.OhGroup(instances=True)

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
        self.assertTrue(np.array_equal(tmp, self.group.tcheck1))

    def test_check2(self):
        tmp = [1, 6, 6, 6, 8, 8, 12, 1]
        tmp = np.diag(np.ones((8,))*48*np.asarray(tmp))
        self.assertTrue(np.array_equal(tmp, self.group.tcheck2))

if __name__ == "__main__":
    unittest.main(verbosity=2)
