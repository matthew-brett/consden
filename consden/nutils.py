""" Utilities to make it easier to work with nipy functions
"""

import numpy as np

from scipy.linalg import toeplitz

from nipy.modalities.fmri.design import natural_spline
from nipy.algorithms.statistics.formula.formulae import make_recarray

EPS = np.finfo(float).eps

# T1 for gray matter at 3T, in seconds. Wansapura et al. JMRI 9: 531-538 (1999)
T1_GRAY_3T = 1.3
# T1 for gray matter at 1.5T, in seconds.
# https://en.wikipedia.org/wiki/Relaxation_(NMR)#T1
T1_GRAY_1p5T = 0.920


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


def dct_ii_basis(volume_times, order=None):
    """ DCT II basis up to order `order`

    See: https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II

    Basis not normalized to length 1, and therefore, basis is not orthogonal.

    Parameters
    ----------
    volume_times : array-like
        Times of acquisition of each volume.  Must be regular and continuous
        otherwise we raise an error.
    order : int, optional
        Order of DCT-II basis

    Returns
    -------
    dct_basis : array
        Shape ``(len(volume_times), order)`` array with DCT-II basis up to
        order `order`.
    """
    N = len(volume_times)
    if order is None:
        order = N
    if not np.allclose(np.diff(np.diff(volume_times)), 0):
        raise RuntimeError("DCT basis assumes continuous regular sampling")
    n = np.arange(N)
    cycle = np.pi * (n + 0.5) / N
    dct_basis = np.zeros((N, order))
    for k in range(0, order):
        dct_basis[:, k] = np.cos(cycle * k)
    return dct_basis


def openfmri2nipy(fname):
    """ Return contents of OpenFMRI stimulus file `fname` as nipy recarray

    Parameters
    ----------
    fname : str
        Path to OpenFMRI stimulus file

    Returns
    -------
    block_spec : array
        Structured array with fields "start" (corresponding to onset time),
        "end" (onset time plus duration), "amplitude".
    """
    onsets, durations, amplitudes = np.loadtxt(fname).T
    return make_recarray(
        np.column_stack((onsets, onsets + durations, amplitudes)),
        names=['start', 'end', 'amplitude'])
