""" Utilities to make it easier to work with nipy function
"""

import numpy as np

from scipy.linalg import toeplitz

from nipy.modalities.fmri.design import natural_spline
from nipy.algorithms.statistics.formula.formulae import make_recarray
from nipy.modalities.fmri.design_matrix import DesignMatrix

EPS = np.finfo(float).eps


def spline_basis(volume_times, order=3):
    """ Return spline drift basis, with knots in time center
    """
    knots = [np.mean(volume_times)]
    splines = natural_spline(volume_times, order=order, knots=knots)
    return splines / np.max(splines, axis=0)


def delta_basis(volume_times):
    """ Basis containing delta at each volume
    """
    return np.eye(len(volume_times))


def step_basis(volume_times):
    """ Basis containing step function at each volume
    """
    N = len(volume_times)
    return toeplitz(np.ones(N), np.zeros(N))


def t1_basis(volume_times, t1):
    """ Basis for T1 decay with time constant `t1`

    Parameters
    ----------
    volume_times : sequence
        Sequence length ``N`` giving onset times of each volume in seconds.
    t1 : float
        T1 time constant for T1 decay

    Returns
    -------
    t1s : array shape (N, N)
        basis giving T1 decay model for each volume in the columns (t1 decay
        for volume $i$ in ``t1s[:, i]``.
    """
    N = len(volume_times)
    times_since = np.outer(volume_times, np.ones(N)) - volume_times
    return np.where(times_since < 0, 0, np.e ** (-times_since / t1))


def drop_colin(design, tol=None):
    """ Drop colinear columns fron 2D array `design`

    Parameters
    ----------
    design : 2D array-like
    tol : None or float, optional
        Columns declared colinear if abs of cosine between two columns is
        between ``(1 - tol, 1 + tol)``.

    Returns
    -------
    reduced : 2D array-like
        `design` with colinear columns dropped.  If columns ``i`` and ``j`` are
        colinear, we drop column ``j`` if j > i.
    """
    design = np.array(design)
    normed = design / np.sqrt(np.sum(design ** 2, axis=0))
    if tol is None:
        tol = np.finfo(normed.dtype).eps * design.shape[0]
    cosines = normed.T.dot(normed)
    colinear = np.abs((np.abs(cosines) - 1)) < tol
    colin_cols = np.any(np.triu(colinear, 1), axis=0)
    return design[:, ~colin_cols]


def openfmri2nipy(fname):
    """ Return contents of OpenFMRI stimulus `fname` as nipy recarray
    """
    onsets, durations, amplitudes = np.loadtxt(fname).T
    return make_recarray(
        np.column_stack((onsets, onsets + durations, amplitudes)),
        names=['start', 'end', 'amplitude'])
