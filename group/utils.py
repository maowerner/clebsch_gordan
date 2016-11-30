"""Helper functions used in the package"""

import unittest
import numpy as np
import scipy

def clean_complex(data, prec=1e-6):
    _data = np.asarray(data)
    for x in np.nditer(_data, op_flags=["readwrite"]):
        tmp = x.copy()
        if np.abs(x) < prec:
            tmp = complex(0.,0.)
        else:
            if np.abs(x.real) < prec:
                tmp = complex(0., x.imag)
            if np.abs(x.imag) < prec:
                tmp = complex(x.real, 0.)
        # reset and set the value of x
        x[...] *= 0.
        x[...] += tmp
    return _data

def clean_real(data, prec=1e-6):
    _data = np.asarray(data)
    for x in np.nditer(_data, op_flags=["readwrite"]):
        if np.abs(x) < prec:
            x[...] *= 0.
    return _data

def _eq(data1, data2=None, prec=1e-6):
    if data2 is None:
        return np.all(np.abs(data1) < prec)
    else:
        return np.all(np.abs(data1-data2) < prec)

def check_array(d1, d2, msg=None):
    c = np.isclose(d1,d2)
    if not np.all(c):
        if msg is not None:
            raise unittest.TestCase.failureException(msg)
        else:
            string = ["Arrays are not close elementwise.\n\n",
                "array 1:\n%r\n\narray 2:\n%r" % (d1,d2)]
            raise unittest.TestCase.failureException("".join(string))

def binomial(n, k):
    return scipy.misc.comb(n, k)
