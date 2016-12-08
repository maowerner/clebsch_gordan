"""Generators for group representatives, based on quaternions."""

import numpy as np

import utils
import quat

_pmatrix = np.asarray(
        [[[1., 0.], [0., 1.]],
         [[0., 1.], [1., 0.]],
         [[0., -1.j], [1.j, 0.]],
         [[1., 0.], [0., -1.]]], dtype=complex)

def genJ0(elements, inv=False):
    res = np.zeros((len(elements), 1, 1), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(0) * el.i
    else:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(0)
    return res

def genJ1_2(elements, inv=False):
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(0.5) * el.i
    else:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(0.5)
    return res

def genJ1(elements, inv=False):
    res = np.zeros((len(elements), 3, 3), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(1) * el.i
    else:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(1)
    return res

def genJ3_2(elements, inv=False):
    res = np.zeros((len(elements), 4, 4), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(1.5) * el.i
    else:
        for i, el in enumerate(elements):
            res[i] = el.R_matrix(1.5)
    return res

def gen1D(elements, inv=False):
    if not inv:
        return np.ones((len(elements), 1, 1), dtype=complex)
    else:
        res = np.ones((len(elements), 1, 1), dtype=complex)
        for i, el in enumerate(elements):
            res[i] *= el.i
        return res

def gen2D(elements, inv=False):
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            tmp = np.asarray(
                    [[complex(el.q[0], -el.q[3]), complex(-el.q[2], -el.q[1])],
                     [complex(el.q[2], -el.q[1]), complex( el.q[0],  el.q[3])]],
                    dtype=complex)
            res[i] = tmp.copy() * el.i
    else:
        for i, el in enumerate(elements):
            tmp = np.asarray(
                    [[complex(el.q[0], -el.q[3]), complex(-el.q[2], -el.q[1])],
                     [complex(el.q[2], -el.q[1]), complex( el.q[0],  el.q[3])]],
                    dtype=complex)
            res[i] = tmp.copy()
    return res

def gen3D(elements, inv=False):
    res = np.zeros((len(elements), 3, 3), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            res[i] = el.rotation_matrix() * el.i
    else:
        for i, el in enumerate(elements):
            res[i] = el.rotation_matrix()
    return res

def gen4D(elements, inv=False):
    res = np.zeros((len(elements), 4, 4), dtype=complex)
    if inv:
        for i, el in enumerate(elements):
            tmp = np.asarray(
                    [[el.q[0], -el.q[1], -el.q[2], -el.q[3]],
                     [el.q[1],  el.q[0], -el.q[3],  el.q[2]],
                     [el.q[2],  el.q[3],  el.q[0], -el.q[1]],
                     [el.q[3], -el.q[2],  el.q[1],  el.q[0]]], dtype=complex)
            res[i] = tmp.copy() * el.i
    else:
        for i, el in enumerate(elements):
            tmp = np.asarray(
                    [[el.q[0], -el.q[1], -el.q[2], -el.q[3]],
                     [el.q[1],  el.q[0], -el.q[3],  el.q[2]],
                     [el.q[2],  el.q[3],  el.q[0], -el.q[1]],
                     [el.q[3], -el.q[2],  el.q[1],  el.q[0]]], dtype=complex)
            res[i] = tmp.copy()
    return res

# special cases
def compare_quat(elem):
    for i, q in enumerate(quat.qPar):
        if elem.comp(q) or (-elem).comp(q):
            return i
    raise RuntimeError("element not identified: %r" % (elem.str()))

def genEpCMF(elements, inv=False):
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    E  = complex( -0.5, 0.5*np.sqrt(3.0) ) # exp(2Pi/3 i)
    Ec = complex( -0.5,-0.5*np.sqrt(3.0) )
    O  = complex(  0.0, 0.0 )
    I  = complex(  1.0, 0.0 )
    m1 = np.asarray([[ I , O ],
                     [ O , I ]], dtype=complex)
    m2 = np.asarray([[ E , O ],
                     [ O , Ec]], dtype=complex)
    m3 = np.asarray([[ Ec, O ],
                     [ O , E ]], dtype=complex)
    m4 = np.asarray([[ O , I ],
                     [ I , O ]], dtype=complex)
    m5 = np.asarray([[ O , E ],
                     [ Ec, O ]], dtype=complex)
    m6 = np.asarray([[ O , Ec],
                     [ E , O ]], dtype=complex)
    m1list = [x+y*24 for x in [ 0, 1, 2, 3] for y in range(4)]
    m2list = [x+y*24 for x in [ 4, 5, 6, 7] for y in range(4)]
    m3list = [x+y*24 for x in [ 8, 9,10,11] for y in range(4)]
    m4list = [x+y*24 for x in [14,17,18,19] for y in range(4)]
    m5list = [x+y*24 for x in [13,16,20,22] for y in range(4)]
    m6list = [x+y*24 for x in [12,15,21,23] for y in range(4)]
    for i, elem in enumerate(elements):
        r = compare_quat(elem)
        if r in m1list:
            res[i] = m1
        elif r in m2list:
            res[i] = m2
        elif r in m3list:
            res[i] = m3
        elif r in m4list:
            res[i] = m4
        elif r in m5list:
            res[i] = m5
        elif r in m6list:
            res[i] = m6
        else:
            raise RuntimeError("element not identified")
    return res

def genEpMF1(elements, inv=False):
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    O  = complex(  0.0, 0.0 )
    I  = complex(  1.0, 0.0 )
    J  = complex(  0.0, 1.0 )
    m1 = np.asarray([[ I, O],
                     [ O, I]], dtype=complex)
    m2 = np.asarray([[-I, O],
                     [ O,-I]], dtype=complex)
    m3 = np.asarray([[ O, I],
                     [ I, O]], dtype=complex)
    m4 = np.asarray([[ O,-I],
                     [-I, O]], dtype=complex)
    m5 = np.asarray([[ J, O],
                     [ O,-J]], dtype=complex)
    m6 = np.asarray([[-J, O],
                     [ O, J]], dtype=complex)
    m7 = np.asarray([[ O,-J],
                     [ J, O]], dtype=complex)
    m8 = np.asarray([[ O, J],
                     [-J, O]], dtype=complex)
    # hard-coded because I don't know which elements
    # of qPar contribute
    m1list = [x+y*8 for x in [0] for y in range(4)]
    m2list = [x+y*8 for x in [3] for y in range(4)]
    m3list = [x+y*8 for x in [1] for y in range(4)]
    m4list = [x+y*8 for x in [2] for y in range(4)]
    m5list = [x+y*8 for x in [4] for y in range(4)]
    m6list = [x+y*8 for x in [5] for y in range(4)]
    m7list = [x+y*8 for x in [7] for y in range(4)]
    m8list = [x+y*8 for x in [6] for y in range(4)]
    for i, elem in enumerate(elements):
        if i in m1list:
            res[i] = m1
        elif i in m2list:
            res[i] = m2
        elif i in m3list:
            res[i] = m3
        elif i in m4list:
            res[i] = m4
        elif i in m5list:
            res[i] = m5
        elif i in m6list:
            res[i] = m6
        elif i in m7list:
            res[i] = m7
        elif i in m8list:
            res[i] = m8
        else:
            print("what to do with %d" % i)
    return res

if __name__ == "__main__":
    print("for checks execute the test script")

