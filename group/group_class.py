"""Class for groups and their representatives."""

import numpy as np
import sympy
import scipy.misc

import utils

# lookup tables for the different groups
listclasses = [["I", "6C4", "6C8p", "6C8", "8C6", "8C3", "12C4p", "J"],
        ["I", "2C4", "2C8p", "2C8", "4IC4", "4IC4p", "J"],
        ["I", "2C4p", "2IC4", "2IC4p", "J"],
        ["I", "2C6", "2C3", "3IC4", "3IC4p", "J"]]

listirreps = [["A1", "A2", "T1", "T2", "E", "G1", "G2", "H"],
        ["A1", "A2", "B1", "B2", "E", "G1", "G2"],
        ["A1", "A2", "B1", "B2", "G1"],
        ["A1", "A2", "K1", "K2", "E", "G1"]]

listrot = [range(48),
        [0, 3, 6, 9, 12, 15, 18, 1, 2, 4, 5, 37, 38, 43, 44, 47],
        [0, 37, 43, 3, 6, 38, 44, 47],
        [0, 19, 23, 27, 31, 36, 44, 46, 38, 40, 42, 47]]

# multiplication table of the faithful G1 irrep of the
# double octahedral group (from Mathematica)
tcheck = np.asarray( 
           [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], 
            [1, 47, 3, 5, 0, 6, 2, 13, 39, 38, 7, 46, 37, 16, 40, 43, 10, 45, 44, 30, 22, 24, 32, 25, 33, 27, 19, 29, 21, 23, 31, 26, 34, 28, 20, 36, 41, 15, 18, 17, 11, 42, 35, 12, 9, 8, 14, 4], 
            [2, 6, 47, 1, 3, 0, 4, 42, 14, 37, 35, 8, 44, 41, 17, 38, 36, 11, 43, 33, 27, 19, 25, 20, 26, 34, 28, 32, 30, 22, 24, 21, 23, 31, 29, 13, 7, 18, 12, 46, 39, 10, 16, 9, 15, 40, 45, 5], 
            [3, 2, 4, 47, 5, 1, 0, 35, 40, 15, 36, 39, 9, 42, 45, 18, 41, 46, 12, 28, 29, 30, 27, 22, 19, 20, 21, 34, 31, 32, 33, 24, 25, 26, 23, 16, 13, 44, 37, 14, 17, 7, 10, 38, 43, 11, 8, 6], 
            [4, 0, 6, 2, 47, 3, 5, 10, 45, 44, 16, 40, 43, 7, 46, 37, 13, 39, 38, 26, 34, 28, 20, 29, 21, 23, 31, 25, 33, 27, 19, 30, 22, 24, 32, 42, 35, 12, 9, 8, 14, 36, 41, 15, 18, 17, 11, 1], 
            [5, 3, 0, 4, 6, 47, 1, 36, 11, 43, 41, 17, 38, 35, 8, 44, 42, 14, 37, 21, 23, 31, 29, 32, 30, 22, 24, 20, 26, 34, 28, 33, 27, 19, 25, 10, 16, 9, 15, 40, 45, 13, 7, 18, 12, 46, 39, 2], 
            [6, 5, 1, 0, 2, 4, 47, 41, 46, 12, 42, 45, 18, 36, 39, 9, 35, 40, 15, 24, 25, 26, 23, 34, 31, 32, 33, 22, 19, 20, 21, 28, 29, 30, 27, 7, 10, 38, 43, 11, 8, 16, 13, 44, 37, 14, 17, 3], 
            [7, 13, 35, 36, 10, 41, 42, 1, 19, 22, 0, 24, 25, 47, 28, 29, 4, 31, 34, 39, 9, 11, 38, 12, 46, 37, 8, 15, 40, 43, 17, 45, 18, 14, 44, 3, 5, 27, 32, 30, 21, 6, 2, 23, 20, 26, 33, 16], 
            [8, 46, 14, 39, 40, 11, 45, 25, 2, 19, 20, 0, 26, 32, 47, 30, 29, 5, 31, 37, 35, 9, 7, 10, 12, 42, 44, 13, 15, 36, 38, 43, 41, 18, 16, 27, 22, 33, 24, 1, 3, 23, 34, 21, 28, 4, 6, 17], 
            [9, 37, 44, 15, 43, 38, 12, 19, 20, 3, 21, 22, 0, 33, 34, 47, 31, 32, 6, 35, 40, 36, 39, 11, 7, 8, 10, 14, 16, 17, 13, 41, 46, 42, 45, 28, 30, 2, 1, 27, 29, 24, 26, 5, 4, 23, 25, 18], 
            [10, 7, 42, 35, 16, 36, 41, 0, 26, 20, 4, 21, 23, 1, 33, 27, 47, 30, 32, 8, 44, 40, 9, 43, 11, 12, 45, 37, 14, 15, 39, 17, 38, 46, 18, 2, 3, 25, 22, 19, 28, 5, 6, 29, 34, 31, 24, 13], 
            [11, 39, 8, 40, 45, 17, 46, 22, 0, 21, 23, 5, 24, 27, 2, 28, 34, 47, 33, 9, 10, 43, 36, 41, 38, 7, 12, 35, 44, 16, 15, 18, 13, 37, 42, 20, 29, 19, 30, 3, 4, 32, 25, 31, 26, 6, 1, 14], 
            [12, 38, 37, 9, 44, 43, 18, 24, 25, 0, 26, 23, 6, 30, 27, 3, 28, 29, 47, 7, 8, 10, 11, 45, 41, 46, 42, 39, 35, 40, 36, 16, 17, 13, 14, 19, 21, 1, 5, 22, 20, 31, 33, 4, 2, 34, 32, 15], 
            [13, 16, 36, 41, 7, 42, 35, 47, 30, 32, 1, 33, 27, 4, 21, 23, 0, 26, 20, 17, 38, 46, 18, 37, 14, 15, 39, 43, 11, 12, 45, 8, 44, 40, 9, 5, 6, 29, 34, 31, 24, 2, 3, 25, 22, 19, 28, 10], 
            [14, 45, 17, 46, 39, 8, 40, 34, 47, 33, 27, 2, 28, 23, 5, 24, 22, 0, 21, 18, 13, 37, 42, 35, 44, 16, 15, 41, 38, 7, 12, 9, 10, 43, 36, 32, 25, 31, 26, 6, 1, 20, 29, 19, 30, 3, 4, 11], 
            [15, 44, 43, 18, 38, 37, 9, 28, 29, 47, 30, 27, 3, 26, 23, 6, 24, 25, 0, 16, 17, 13, 14, 39, 35, 40, 36, 45, 41, 46, 42, 7, 8, 10, 11, 31, 33, 4, 2, 34, 32, 19, 21, 1, 5, 22, 20, 12], 
            [16, 10, 41, 42, 13, 35, 36, 4, 31, 34, 47, 28, 29, 0, 24, 25, 1, 19, 22, 45, 18, 14, 44, 15, 40, 43, 17, 12, 46, 37, 8, 39, 9, 11, 38, 6, 2, 23, 20, 26, 33, 3, 5, 27, 32, 30, 21, 7], 
            [17, 40, 11, 45, 46, 14, 39, 29, 5, 31, 32, 47, 30, 20, 0, 26, 25, 2, 19, 43, 41, 18, 16, 13, 15, 36, 38, 10, 12, 42, 44, 37, 35, 9, 7, 23, 34, 21, 28, 4, 6, 27, 22, 33, 24, 1, 3, 8], 
            [18, 43, 38, 12, 37, 44, 15, 31, 32, 6, 33, 34, 47, 21, 22, 0, 19, 20, 3, 41, 46, 42, 45, 14, 16, 17, 13, 11, 7, 8, 10, 35, 40, 36, 39, 24, 26, 5, 4, 23, 25, 28, 30, 2, 1, 27, 29, 9], 
            [19, 33, 28, 30, 21, 24, 26, 37, 35, 39, 9, 7, 8, 18, 16, 17, 43, 41, 45, 27, 3, 22, 1, 0, 25, 2, 20, 47, 29, 5, 32, 23, 6, 34, 4, 15, 38, 14, 46, 13, 36, 12, 44, 11, 40, 10, 42, 31], 
            [20, 25, 34, 27, 29, 22, 23, 8, 44, 35, 40, 9, 10, 46, 18, 13, 17, 38, 41, 2, 28, 3, 19, 21, 0, 26, 4, 33, 47, 30, 1, 5, 24, 6, 31, 14, 39, 42, 7, 37, 15, 11, 45, 36, 16, 43, 12, 32], 
            [21, 19, 26, 28, 31, 30, 24, 9, 10, 40, 43, 36, 11, 37, 42, 14, 18, 13, 46, 20, 4, 29, 3, 5, 22, 0, 23, 2, 34, 47, 27, 32, 1, 25, 6, 44, 15, 8, 39, 35, 16, 38, 12, 17, 45, 41, 7, 33], 
            [22, 27, 20, 29, 23, 32, 25, 39, 9, 36, 11, 38, 7, 14, 44, 16, 45, 18, 42, 3, 21, 5, 30, 24, 1, 19, 0, 28, 4, 31, 47, 6, 33, 2, 26, 40, 17, 35, 13, 15, 43, 46, 8, 41, 10, 12, 37, 34], 
            [23, 22, 25, 20, 34, 29, 32, 11, 12, 10, 45, 43, 41, 39, 37, 35, 14, 15, 13, 0, 26, 4, 21, 31, 5, 24, 6, 19, 2, 28, 3, 47, 30, 1, 33, 8, 40, 7, 36, 9, 44, 17, 46, 16, 42, 18, 38, 27], 
            [24, 30, 19, 21, 26, 31, 33, 38, 7, 11, 12, 41, 46, 15, 35, 40, 44, 16, 14, 22, 0, 23, 5, 6, 32, 1, 25, 3, 20, 4, 29, 34, 47, 27, 2, 9, 43, 39, 17, 36, 10, 18, 37, 45, 8, 42, 13, 28], 
            [25, 32, 27, 22, 20, 23, 34, 46, 37, 7, 8, 12, 42, 17, 15, 36, 40, 43, 16, 1, 19, 0, 24, 26, 6, 33, 2, 30, 3, 21, 5, 4, 31, 47, 28, 39, 11, 13, 41, 38, 9, 45, 14, 10, 35, 44, 18, 29], 
            [26, 24, 33, 19, 28, 21, 31, 12, 42, 8, 44, 10, 45, 38, 13, 39, 15, 36, 17, 25, 2, 20, 0, 4, 23, 6, 34, 1, 27, 3, 22, 29, 5, 32, 47, 37, 9, 46, 11, 7, 35, 43, 18, 40, 14, 16, 41, 30], 
            [27, 34, 29, 32, 22, 25, 20, 14, 15, 13, 39, 37, 35, 45, 43, 41, 11, 12, 10, 47, 30, 1, 33, 19, 2, 28, 3, 31, 5, 24, 6, 0, 26, 4, 21, 17, 46, 16, 42, 18, 38, 8, 40, 7, 36, 9, 44, 23], 
            [28, 26, 31, 33, 30, 19, 21, 44, 16, 14, 15, 35, 40, 12, 41, 46, 38, 7, 11, 34, 47, 27, 2, 3, 20, 4, 29, 6, 32, 1, 25, 22, 0, 23, 5, 18, 37, 45, 8, 42, 13, 9, 43, 39, 17, 36, 10, 24], 
            [29, 20, 23, 34, 32, 27, 22, 40, 43, 16, 17, 15, 36, 8, 12, 42, 46, 37, 7, 4, 31, 47, 28, 30, 3, 21, 5, 26, 6, 33, 2, 1, 19, 0, 24, 45, 14, 10, 35, 44, 18, 39, 11, 13, 41, 38, 9, 25], 
            [30, 28, 21, 31, 24, 33, 19, 15, 36, 17, 38, 13, 39, 44, 10, 45, 12, 42, 8, 29, 5, 32, 47, 1, 27, 3, 22, 4, 23, 6, 34, 25, 2, 20, 0, 43, 18, 40, 14, 16, 41, 37, 9, 46, 11, 7, 35, 26], 
            [31, 21, 24, 26, 33, 28, 30, 43, 41, 45, 18, 16, 17, 9, 7, 8, 37, 35, 39, 23, 6, 34, 4, 47, 29, 5, 32, 0, 25, 2, 20, 27, 3, 22, 1, 12, 44, 11, 40, 10, 42, 15, 38, 14, 46, 13, 36, 19], 
            [32, 29, 22, 23, 25, 34, 27, 17, 38, 41, 46, 18, 13, 40, 9, 10, 8, 44, 35, 5, 24, 6, 31, 33, 47, 30, 1, 21, 0, 26, 4, 2, 28, 3, 19, 11, 45, 36, 16, 43, 12, 14, 39, 42, 7, 37, 15, 20], 
            [33, 31, 30, 24, 19, 26, 28, 18, 13, 46, 37, 42, 14, 43, 36, 11, 9, 10, 40, 32, 1, 25, 6, 2, 34, 47, 27, 5, 22, 0, 23, 20, 4, 29, 3, 38, 12, 17, 45, 41, 7, 44, 15, 8, 39, 35, 16, 21], 
            [34, 23, 32, 25, 27, 20, 29, 45, 18, 42, 14, 44, 16, 11, 38, 7, 39, 9, 36, 6, 33, 2, 26, 28, 4, 31, 47, 24, 1, 19, 0, 3, 21, 5, 30, 46, 8, 41, 10, 12, 37, 40, 17, 35, 13, 15, 43, 22], 
            [35, 42, 16, 13, 36, 7, 10, 2, 28, 27, 3, 19, 20, 6, 31, 32, 5, 24, 23, 14, 15, 39, 37, 9, 8, 44, 40, 18, 17, 38, 46, 11, 12, 45, 43, 47, 1, 34, 25, 33, 30, 0, 4, 22, 29, 21, 26, 41], 
            [36, 35, 10, 16, 41, 13, 7, 3, 21, 29, 5, 30, 22, 2, 26, 34, 6, 33, 25, 40, 43, 17, 15, 38, 39, 9, 11, 44, 45, 18, 14, 46, 37, 8, 12, 4, 47, 20, 27, 28, 31, 1, 0, 32, 23, 24, 19, 42], 
            [37, 18, 15, 38, 9, 12, 44, 33, 27, 1, 19, 25, 2, 31, 29, 5, 21, 23, 4, 13, 39, 7, 46, 8, 42, 14, 35, 17, 36, 11, 41, 10, 45, 16, 40, 30, 24, 47, 6, 32, 22, 26, 28, 0, 3, 20, 34, 43], 
            [38, 15, 9, 43, 12, 18, 37, 30, 22, 5, 24, 32, 1, 28, 20, 4, 26, 34, 2, 36, 11, 41, 17, 46, 13, 39, 7, 40, 10, 45, 16, 42, 14, 35, 8, 21, 31, 3, 47, 29, 23, 33, 19, 6, 0, 25, 27, 44], 
            [39, 14, 40, 17, 11, 46, 8, 27, 3, 30, 22, 1, 19, 34, 4, 31, 23, 6, 26, 15, 36, 38, 13, 7, 37, 35, 9, 16, 43, 41, 18, 12, 42, 44, 10, 29, 32, 28, 33, 47, 5, 25, 20, 24, 21, 0, 2, 45], 
            [40, 8, 45, 14, 17, 39, 11, 20, 4, 28, 29, 3, 21, 25, 6, 33, 32, 1, 24, 44, 16, 15, 35, 36, 9, 10, 43, 42, 18, 13, 37, 38, 7, 12, 41, 34, 27, 26, 19, 2, 47, 22, 23, 30, 31, 5, 0, 46], 
            [41, 36, 7, 10, 42, 16, 13, 5, 24, 23, 6, 31, 32, 3, 19, 20, 2, 28, 27, 11, 12, 45, 43, 18, 17, 38, 46, 9, 8, 44, 40, 14, 15, 39, 37, 0, 4, 22, 29, 21, 26, 47, 1, 34, 25, 33, 30, 35], 
            [42, 41, 13, 7, 35, 10, 16, 6, 33, 25, 2, 26, 34, 5, 30, 22, 3, 21, 29, 46, 37, 8, 12, 44, 45, 18, 14, 38, 39, 9, 11, 40, 43, 17, 15, 1, 0, 32, 23, 24, 19, 4, 47, 20, 27, 28, 31, 36], 
            [43, 9, 12, 44, 18, 15, 38, 21, 23, 4, 31, 29, 5, 19, 25, 2, 33, 27, 1, 10, 45, 16, 40, 17, 36, 11, 41, 8, 42, 14, 35, 13, 39, 7, 46, 26, 28, 0, 3, 20, 34, 30, 24, 47, 6, 32, 22, 37], 
            [44, 12, 18, 37, 15, 9, 43, 26, 34, 2, 28, 20, 4, 24, 32, 1, 30, 22, 5, 42, 14, 35, 8, 40, 10, 45, 16, 46, 13, 39, 7, 36, 11, 41, 17, 33, 19, 6, 0, 25, 27, 21, 31, 3, 47, 29, 23, 38], 
            [45, 11, 46, 8, 14, 40, 17, 23, 6, 26, 34, 4, 31, 22, 1, 19, 27, 3, 30, 12, 42, 44, 10, 16, 43, 41, 18, 7, 37, 35, 9, 15, 36, 38, 13, 25, 20, 24, 21, 0, 2, 29, 32, 28, 33, 47, 5, 39], 
            [46, 17, 39, 11, 8, 45, 14, 32, 1, 24, 25, 6, 33, 29, 3, 21, 20, 4, 28, 38, 7, 12, 41, 42, 18, 13, 37, 36, 9, 10, 43, 44, 16, 15, 35, 22, 23, 30, 31, 5, 0, 34, 27, 26, 19, 2, 47, 40], 
            [47, 4, 5, 6, 1, 2, 3, 16, 17, 18, 13, 14, 15, 10, 11, 12, 7, 8, 9, 31, 32, 33, 34, 27, 28, 29, 30, 23, 24, 25, 26, 19, 20, 21, 22, 41, 42, 43, 44, 45, 46, 35, 36, 37, 38, 39, 40, 0]],
           dtype=int)

# local functions
def _get_numbers(name):
    if name in ["I", "J"]:
        return 1
    elif name[:2].isdigit():
        return int(name[:2])
    else:
        return int(name[0])

def _get_instance(name, p2=0):
    try:
        if name == "A1":
            cls = OhA1(p2=p2)
        elif name == "A2":
            cls = OhA2(p2=p2)
        elif name == "B1":
            cls = OhB1(p2=p2)
        elif name == "B2":
            cls = OhB2(p2=p2)
        elif name == "K1":
            cls = OhK1(p2=p2)
        elif name == "K2":
            cls = OhK2(p2=p2)
        elif name == "T1":
            cls = OhT1(p2=p2)
        elif name == "T2":
            cls = OhT2(p2=p2)
        elif name == "G1":
            cls = OhG1(p2=p2)
        elif name == "G2":
            cls = OhG2(p2=p2)
        elif name == "E":
            cls = OhE(p2=p2)
        elif name == "H":
            cls = OhH(p2=p2)
    except AttributeError:
        print("There is no %d irrep known" % name)
        sys.exit(-1)
    return cls

class OhGroup(object):
    """Basic octahedral group implementation,
    based on code by B. C. Metsch.
    """

    def __init__(self, p2=0, dimension=None, instances=False, prec=1e-6):
        """Initialize the group.

        Parameters
        ----------
        p2 : int, optional
            The total momentum squared of the momentum described.
        dimension : int or None, optional
            The dimension of the irrep, defaults to None
        instances : bool, optional
            Instanciate all subgroups of the group, defaults to False.
        prec : float, optional
            The precision for comparisons.
        """
        # save the arguments
        self.prec = prec
        self.p2 = p2

        # p2 > 3 not implemented
        if p2 > 3:
            raise NotImplementedError("P^2 > 3 not implemented!")

        # look up the members
        self.lclasses = listclasses[p2]
        self.lirreps = listirreps[p2]
        self.lrotations = listrot[p2]

        # the number of elements in each class
        self.sclass = [_get_number(x) for x in self.lclasses]

        # representative element of each class
        tmp = list(np.cumsum(self.sclass))
        self.rclass = [self.lrotations[j-self.sclass[i]] for i, j in enumerate(tmp)]

        # the number of elements in group, classes and irreps
        self.order = len(self.lrotations)
        self.nclasses = len(self.lclasses)
        self.nirreps = len(self.lirreps)

        # tables for classes, characters, mutliplications table and checks
        self.tclass = np.zeros((self.order, self.order), dtype=int)
        self.tmult = np.zeros((self.order, self.order), dtype=int)

        self.tchar = np.zeros((self.nirreps, self.nclasses), dtype=complex)
        self.tcheck1 = np.zeros((self.nirreps, self.nirreps), dtype=complex)
        self.tcheck1.fill(-1.)
        self.tcheck2 = np.zeros((self.nclasses, self.nclasses), dtype=complex)
        self.tcheck2.fill(-1.)

        self.instances = None
        if instances:
            self.instances = [_get_instances(name, self.p2) for name in self.lirreps]

            # fill the character table
            for ir in self.instances:
                self.InsertCharTableRow(ir)

            # do checks, need the character table
            check1 = self.Check1()
            check2 = self.Check2()
            if not check1:
                print("Check 1 failed")
            if not check2:
                print("Check 2 failed")

        # important for irreps
        self.dim = dimension
        self.irid = -1
        self.mx = None
        if self.dim is not None:
            self.mx = np.zeros((self.order, self.dim, self.dim), dtype=complex)
        self.faithful = False

    def Inverse(self, index):
        return np.nonzero(self.tmult[index] == 0)[0][0]

    def InsertCharTableRow(self, irrep):
        i = irrep.irid
        for k in range(self.nclasses):
            l = self.rclass[k]
            elem = irrep.mx(self.lrotations.index(l))
            tr = utils.clean_complex(np.trace(mx), self.prec)
            self.tchar[i,k] = tr

    def MkClassTable(self):
        for m in range(self.order):
            for l in range(self.order):
                for k in range(self.order):
                    kinv = self.Inverse(k)
                    gkl = self.tmult[k,l]
                    gklkinv = self.tmult[gkl,kinv]
                    if gklkinv == m:
                        self.tclass[m,l] = m
                        break

    def MkMultTbl(self):
        mt = np.zeros((self.irorder, self.irorder), dtype=int)
        for i in range(self.irorder):
            mxi = self.mx[i]
            for j in range(self.irorder):
                mxj = self.mx[j]
                prodij = np.dot(mxi, mxj)
                for k in range(self.irorder):
                    mxk = self.mx[k]
                    if utils._eq(prodij, mxk, self.prec):
                        mt[i,j] = k
                    self.tmultir[i,j] = mt[i,j]
        # check if faithful
        if self.dim == 1:
            # 1D irreps are not faithful
            self.faithful = False
        else:
            self.faithful = True
            for i in range(self.irorder):
                row = np.unique(mt[i])
                col = np.unique(mt[:,i])
                if row.size != self.irorder or col.size != self.irorder:
                    self.faithful = False
                    break

    def Check1(self):
        for i in range(self.nirreps):
            for j in range(self.nirreps):
                res = 0.
                for k in range(self.nclasses):
                    res += self.tchar[i,k] * self.tchar[j,k].conj() * self.sclass[k]
                self.tcheck1[i,j] = res
        # check against
        tmp = self.order * np.identity(self.nclasses)
        return utils._eq(self.tcheck1, tmp)

    def Check2(self):
        for i in range(self.nclasses):
            for j in range(self.nclasses):
                res = 0.
                for k in range(self.nirreps):
                    res += self.tchar[k,i] * self.tcheck[k,j].conj()
                self.tcheck[i,j] = res
        # check against
        tmp = np.diag(np.ones((self.nirreps,))*self.order/np.asarray(self.sclass))
        return utils._eq(self.tcheck2, tmp)


if __name__ == "__main__":
    print("for checks execute the test script")

