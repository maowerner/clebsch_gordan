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
    for i, el in enumerate(elements):
        res[i] = el.rotation_matrix(inv)
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
        if elem.comp(q) or elem.comp(np.negative(q)):
            return i
    raise RuntimeError("element not identified: %r" % (elem.str()))

def genEpCMF(elements, inv=False):
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    I  = complex( 1.0, 0.0 )
    O  = complex( 0.0, 0.0 )
    E  = np.exp(2.j*np.pi/3.)
    F = E.conj()
    m1 = np.asarray([[ I, O ],
                     [ O, I ]], dtype=complex)
    m2 = np.asarray([[ E, O ],
                     [ O, F ]], dtype=complex)
    m3 = np.asarray([[ F, O ],
                     [ O, E ]], dtype=complex)
    m4 = np.asarray([[ O, F ],
                     [ E, O ]], dtype=complex)
    m5 = np.asarray([[ O, E ],
                     [ F, O ]], dtype=complex)
    m6 = np.asarray([[ O, I ],
                     [ I, O ]], dtype=complex)
    m1list = [x+y*24 for x in [ 0, 1, 2, 3] for y in range(4)]
    m2list = [x+y*24 for x in [ 4, 5, 6, 7] for y in range(4)]
    m3list = [x+y*24 for x in [ 8, 9,10,11] for y in range(4)]
    m4list = [x+y*24 for x in [12,15,21,23] for y in range(4)]
    m5list = [x+y*24 for x in [13,16,20,22] for y in range(4)]
    m6list = [x+y*24 for x in [14,17,18,19] for y in range(4)]
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

def genEpCMF_old(elements, inv=False):
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    E  = complex( 0.5*np.sqrt(3.0), 0.0 )
    H  = complex( 0.5, 0.0 )
    O  = complex( 0.0, 0.0 )
    I  = complex( 1.0, 0.0 )
    m1 = np.asarray([[ I, O ],
                     [ O, I ]], dtype=complex)
    m2 = np.asarray([[-H, E ],
                     [-E,-H ]], dtype=complex)
    m3 = np.asarray([[-H,-E ],
                     [ E,-H ]], dtype=complex)
    m4 = np.asarray([[ I, O ],
                     [ O,-I ]], dtype=complex)
    m5 = np.asarray([[-H,-E ],
                     [-E, H ]], dtype=complex)
    m6 = np.asarray([[-H, E ],
                     [ E, H ]], dtype=complex)
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
    m1list = [x+y*8 for x in [0] for y in range(2)]
    m2list = [x+y*8 for x in [1] for y in range(2)]
    m3list = [x+y*8 for x in [4] for y in range(2)]
    m4list = [x+y*8 for x in [5] for y in range(2)]
    m5list = [x+y*8 for x in [2] for y in range(2)]
    m6list = [x+y*8 for x in [3] for y in range(2)]
    m7list = [x+y*8 for x in [7] for y in range(2)]
    m8list = [x+y*8 for x in [6] for y in range(2)]
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

def genT1CMF(elements, inv=False):
    res = np.zeros((len(elements), 3, 3), dtype=complex)
    pars = np.zeros((9,), dtype=complex)
    for i, elem in enumerate(elements):
        pars.fill(0.)
        r = compare_quat(elem)
        if r in [0,2,13,16,]:
            pars[0] = 1.
        elif r in [1,3,20,22]:
            pars[0] = -1.
        if r in [8,10,15,21]:
            pars[1] = 1.
        elif r in [9,11,12,23]:
            pars[1] = -1.
        if r in [6,7,17,19]:
            pars[2] = 1.
        elif r in [4,5,14,18]:
            pars[2] = -1.
        if r in [5,7,15,23]:
            pars[3] = 1.
        elif r in [4,6,12,21]:
            pars[3] = -1.
        if r in [0,3,14,17]:
            pars[4] = 1.
        elif r in [1,2,18,19]:
            pars[4] = -1.
        if r in [9,10,13,22]:
            pars[5] = 1.
        elif r in [8,11,16,20]:
            pars[5] = -1.
        if r in [8,9,17,18]:
            pars[6] = 1.
        elif r in [10,11,14,19]:
            pars[6] = -1.
        if r in [5,6,16,22]:
            pars[7] = 1.
        elif r in [4,7,13,20]:
            pars[7] = -1.
        if r in [0,1,12,15]:
            pars[8] = 1.
        elif r in [2,3,21,23]:
            pars[8] = -1.
        res[i][0,0] = pars[0]*1.
        res[i][0,1] = pars[1]*1.j
        res[i][0,2] = pars[2]*1.j
        res[i][1,0] = pars[3]*1.j
        res[i][1,1] = pars[4]*1.
        res[i][1,2] = pars[5]*1.
        res[i][2,0] = pars[6]*1.j
        res[i][2,1] = pars[7]*1.
        res[i][2,2] = pars[8]*1.
        if inv:
            res[i] *= elem.i
    return res

if __name__ == "__main__":
    print("for checks execute the test script")

