"""Generators for group representatives."""

import numpy as np

import utils

# Pauli matrices as generator for 2D representations
_pauli_matrices = [np.identity(2, dtype=complex)]
_pauli_matrices.append(np.asarray([[0., 1.],[1., 0.]], dtype=complex))
_pauli_matrices.append(np.asarray([[0., -1.j],[1.j, 0.]], dtype=complex))
_pauli_matrices.append(np.asarray([[1., 0.],[0., -1.]], dtype=complex))

def gen_G1(rotation):
    prec = 1e-6
    vector = rotation.get_vector()
    omega = rotation.get_omega()
    result = np.cos(omega/2)*_pauli_matrices[0]
    for i, p in enumerate(_pauli_matrices[1:]):
        result -= 1.j*np.sin(omega/2)*vector[i]*p
    return utils.clean_complex(result, prec)

def gen_T1(rotation):
    prec = 1e-6
    # compute a vector rotation:
    # exp(-i omega (n.L) = exp(A[omega n]) where omega is the rotation angle, 
    # L the angular momentum operator and n the (normalised) rotation axis
    # A[omega n] x = omega [ n x x ] (vector product) 
    vector = rotation.get_vector()
    omega = rotation.get_omega()
    x, y, z = vector
    #print(vector)
    #print(omega)
    s = np.sin(omega)
    c = np.cos(omega)
    omc = 1. - c
    result = np.outer(vector, vector)*omc
    tmp = np.asarray([[c, -s*z, s*y], [s*z, c, -s*x], [-s*y, s*x, c]])
    result += tmp
    return utils.clean_real(result, prec)

def gen_H(rotation):
    # compute a spin 3/2 rotation:
    # matrix elements according to 
    # D.A. Varshalivich: Quantum Theory of Angular Momentum, 
    # World Scientific, first reprint 1989, Table 4.25
    prec = 1e-6
    vector = rotation.get_vector()
    omega = rotation.get_omega()
    c = np.cos(omega/2.)
    s = np.sin(omega/2.)
    x, y, z = vector
    xp = x+1.j*y
    xm = x-1.j*y
    sq3 = np.sqrt(3)
    result = np.asarray([[(c - 1.j * s * z)**3,
             -1.j * sq3 * s * xm * ((c - 1.j * s * z)**2), 
             -sq3 * ((s * xm)**2) * (c - 1.j * s * z),
              1.j * ((s * xm)**3)],
            [ -1.j * sq3 * s * xp * ((c - 1.j * s * z)**2),
             ( 1. - 3. * s*s * (x*x + y*y) ) * (c - 1.j * s * z),
             -1.j * s * xm * ( 2. - 3. * s*s * ( x*x + y*y ) ),
             -sq3 * ((s * xm)**2) * (c + 1.j * s * z)],
            [-sq3 * ((s * xp)**2) * (c - 1.j * s * z),
             -1.j * s * xp * (2. - 3. * s*s * (x*x + y*y)),
             ( 1. - 3. * s*s * (x*x + y*y)) * (c + 1.j * s * z),
             -1.j * sq3 * s * xm * ((c + 1.j * s * z)**2)],
            [1.j * ((s * xp)**3),
             -sq3 * ((s * xp)**2) * (c + 1.j * s * z),
             -1.j * sq3 * s * xp * ((c + 1.j * s * z)**2),
             (c + 1.j * s * z)**3]])
    return utils.clean_complex(result, prec)

def gen_E(p2):
    mx = None
    if p2 == 0:
        mx = np.zeros((48, 2, 2), dtype=complex)
        res = np.asarray([[1, 0], [0, 1]])
        for k in range(0, 7):
            mx[k] = res.copy()
        c = np.cos(np.pi/3)
        s = np.sin(np.pi/3)
        res = - c * _pauli_matrices[3] - s * _pauli_matrices[1]
        mx[7] = res.copy()
        res = - c * _pauli_matrices[3] + s * _pauli_matrices[1]
        mx[8] = res.copy()
        res = _pauli_matrices[3]
        mx[9] = res.copy()
        mx[10] = mx[7].copy()
        mx[11] = mx[8].copy()
        mx[12] = mx[9].copy()
        for k in range(13, 19):      
            mx[k] = mx[k-6].copy()
        res = -c*mx[0] - 1.j*s*_pauli_matrices[2]
        mx[19] = res.copy()
        res = -c*mx[0] + 1.j*s*_pauli_matrices[2]
        mx[20] = res.copy()
        mx[21] = mx[19].copy()
        mx[22] = mx[20].copy()
        mx[23] = mx[22].copy()
        mx[24] = mx[21].copy()
        mx[25] = mx[23].copy()
        mx[26] = mx[24].copy()
        mx[27] = mx[25].copy()
        mx[28] = mx[26].copy()
        mx[29] = mx[27].copy()
        mx[30] = mx[28].copy()
        mx[31] = mx[30].copy()
        mx[32] = mx[29].copy()
        mx[33] = mx[31].copy()
        mx[34] = mx[32].copy()
        mx[35] = mx[16].copy()
        mx[36] = mx[35].copy()
        res = _pauli_matrices[3]
        mx[37] = res.copy()
        mx[38] = mx[37].copy()
        mx[39] = mx[17].copy()
        mx[40] = mx[39].copy()
        for k in range(41,47):
            mx[k] = mx[k-6].copy()
        mx[47] = mx[0].copy()
    elif p2 == 1:
        mx = np.zeros((16, 2, 2), dtype=complex)
        mx[0] = _pauli_matrices[0].copy()
        mx[1] = -1.*_pauli_matrices[0]
        mx[2] = mx[1].copy()
        mx[3] = 1.j*_pauli_matrices[3]
        mx[4] = -1.j*_pauli_matrices[3]
        mx[5] = mx[4].copy()
        mx[6] = mx[3].copy()
        mx[7] = _pauli_matrices[1].copy()
        mx[8] = -1.*_pauli_matrices[1]
        mx[9] = mx[7].copy()
        mx[10] = mx[8].copy()
        mx[11] = -1.*_pauli_matrices[2]
        mx[12] = _pauli_matrices[2].copy()
        mx[13] = mx[11].copy()
        mx[14] = mx[12].copy()
        mx[15] = mx[0].copy()
    elif p2 == 3:
        mx = np.zeros((12, 2, 2), dtype=complex)
        sq3 = np.sqrt(3)
        mx[0] = _pauli_matrices[0].copy()
        mx[1] = np.asarray([[-0.5, -0.5*sq3], [0.5*sq3, -0.5]])
        mx[2] = np.asarray([[-0.5, 0.5*sq3], [-0.5*sq3, -0.5]])
        mx[3] = mx[2].copy()
        mx[4] = mx[1].copy()
        mx[5] = np.asarray([[0.5, 0.5*sq3], [0.5*sq3, -0.5]])
        mx[6] = np.asarray([[-1., 0.], [0., 1.]])
        mx[7] = np.asarray([[0.5, -0.5*sq3], [-0.5*sq3, -0.5]])
        mx[8] = mx[6].copy()
        mx[9] = mx[7].copy()
        mx[10] = mx[5].copy()
        mx[11] = mx[0].copy()
    return mx

if __name__ == "__main__":
    print("for checks execute the test script")

