"""Init script to initialize lots of groups and save them to disk."""

import numpy as np

import group_class_quat as gc
import utils

def init_groups(p2max=4, prefs=None, U2=None, U3=None, U4=None,
        path=None, fname=None):
    groups = []

    # check reference vectors
    if prefs is None:
        _prefs = [[0.,0.,0.], [0.,0.,1.],
                  [1.,1.,0.], [1.,1.,1.],
                  [0.,0.,2.], [0.,1.,2.],
                  [1.,1.,2.]]
        if p2max > 6:
            _p2max = 6
        else:
            _p2max = p2max
    else:
        _prefs = prefs
        if p2max > len(_prefs):
            _p2max = len(_prefs)
        else:
            _p2max = p2max

    # check basis change matrices
    if U2 is None:
        _U2 = np.identity(2)
    else:
        _U2 = U2
    if U3 is None:
        _U3 = np.identity(3)
    else:
        _U3 = U3
    if U4 is None:
        _U4 = np.identity(4)
    else:
        _U4 = U4

    for p2 in range(p2max):
        try:
            _g = gc.TOh.read(p2=p2, path=path)
            if (not np.allclose(_g.U4, _U4) or 
                not np.allclose(_g.U3, _U3) or
                not np.allclose(_g.U2, _U2)):
                raise IOError("redo computation")
        except IOError:
            _g = gc.TOh(pref=prefs[p2], irreps=True, U4=_U4, U3=_U3, U2=_U2)
            _g.save(path=path, fname=fname)
        groups.append(_g)
    return groups
