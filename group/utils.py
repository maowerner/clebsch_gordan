"""Helper functions used in the package"""

import numpy as np

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
