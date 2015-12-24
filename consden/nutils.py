""" Utilities to make it easier to work with nipy function
"""

import numpy as np

from scipy.linalg import toeplitz

from nipy.modalities.fmri.design import natural_spline
from nipy.algorithms.statistics.formula.formulae import make_recarray
from nipy.modalities.fmri.design_matrix import DesignMatrix


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


def openfmri2nipy(fname):
    """ Return contents of OpenFMRI stimulus `fname` as nipy recarray
    """
    onsets, durations, amplitudes = np.loadtxt(fname).T
    return make_recarray(
        np.column_stack((onsets, onsets + durations, amplitudes)),
        names=['start', 'end', 'amplitude'])
