"""Generators for group representatives, based on quaternions."""

import numpy as np

import utils

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

if __name__ == "__main__":
    print("for checks execute the test script")

