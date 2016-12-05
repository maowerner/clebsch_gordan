"""Generators for group representatives, based on quaternions."""

import numpy as np

import utils


def genJ0(elements):
    res = np.zeros((len(elements), 1, 1), dtype=complex)
    for i, el in enumerate(elements):
        res[i] = el.R_matrix(0)
    return res

def genJ1_2(elements):
    res = np.zeros((len(elements), 2, 2), dtype=complex)
    for i, el in enumerate(elements):
        res[i] = el.R_matrix(0.5)
    return res

def genJ1(elements):
    res = np.zeros((len(elements), 3, 3), dtype=complex)
    for i, el in enumerate(elements):
        res[i] = el.R_matrix(1)
    return res

def genJ3_2(elements):
    res = np.zeros((len(elements), 4, 4), dtype=complex)
    for i, el in enumerate(elements):
        res[i] = el.R_matrix(1.5)
    return res

if __name__ == "__main__":
    print("for checks execute the test script")

