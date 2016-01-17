""" Utilities to make it easier to work with nipy functions
"""

import numpy as np
import numpy.linalg as npl

from scipy.linalg import toeplitz

from nipy.modalities.fmri.design import natural_spline
from nipy.algorithms.statistics.formula.formulae import make_recarray

EPS = np.finfo(float).eps

# T1 for gray matter at 3T, in seconds. Wansapura et al. JMRI 9: 531-538 (1999)
T1_GRAY_3T = 1.3
# T1 for gray matter at 1.5T, in seconds.
# https://en.wikipedia.org/wiki/Relaxation_(NMR)#T1
T1_GRAY_1p5T = 0.920

# Convert from Tesla to T1 constant in gray matter
TESLA2T1_GRAY = {'1.5': T1_GRAY_1p5T,
                 1.5: T1_GRAY_1p5T,
                 '3.0': T1_GRAY_3T,
                 '3': T1_GRAY_3T,
                 3: T1_GRAY_3T,
                 3.0: T1_GRAY_3T}


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
    r""" Basis for T1 decay with time constant `t1`

    From Wikipedia: https://en.wikipedia.org/wiki/Spin%E2%80%93lattice_relaxation

    .. math::

        M_z(t) = M_{z,\mathrm{eq}}
                 - \left [ M_{z,\mathrm{eq}}
                 - M_{z}(0) \right ] e^{-t/T_1}

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


def _norm_cols(X):
    """ Divide columns of array `X` by Euclidean lengths

    Return normalized `X` and norms
    """
    X = np.asarray(X)
    norms = npl.norm(X, axis=0)
    # Avoid divide by zero warning for zero-length columns
    return X / np.where(norms == 0, 1, norms), norms


def drop_colin(design, ref_design=None, tol=None):
    """ Drop colinear columns from 2D array `design`

    Zero-length columns also dropped.

    Parameters
    ----------
    design : 2D array-like
        Array with columns that may be removed because of colinearity.
    ref_design : None or 2D array-like, optional
        Array containing columns for which we will check colinearity against
        the columns of `design`.  Default of None corresponds to `design`.
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
    normed, norms = _norm_cols(design)
    against_self = ref_design is None
    ref_normed = normed if against_self else _norm_cols(ref_design)[0]
    if tol is None:
        tol = np.finfo(normed.dtype).eps * design.shape[0]
    cosines = ref_normed.T.dot(normed)
    colinear = np.abs((np.abs(cosines) - 1)) < tol
    if against_self:  # Ignore colinearity of column with self
        colinear = np.triu(colinear, 1)
    colin_cols = np.any(colinear, axis=0)
    return design[:, ~colin_cols & (np.abs(norms) > tol)]


def demean_cols(X):
    """ Remove mean from columns of 2D array `X`
    """
    X = np.asarray(X)
    return X - np.mean(X, axis=0)


def dct_ii_basis(volume_times, order=None, normcols=False):
    """ DCT II basis up to order `order`

    See: https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II

    By default, basis not normalized to length 1, and therefore, basis is not
    orthogonal.  Normalize basis with `normcols` keyword argument.

    Parameters
    ----------
    volume_times : array-like
        Times of acquisition of each volume.  Must be regular and continuous
        otherwise we raise an error.
    order : int, optional
        Order of DCT-II basis
    normcols : bool, optional
        If True, normalize columns to length 1, so return orthogonal
        `dct_basis`.

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
    if normcols:  # Set column lengths to 1
        lengths = np.ones(order) * np.sqrt(N / 2.)
        lengths[0:1] = np.sqrt(N)  # Allow order=0
        dct_basis /= lengths
    return dct_basis


def openfmri2nipy(ons_dur_amp):
    """ Contents of OpenFMRI condition file `ons_dur_map` as nipy recarray

    Parameters
    ----------
    ons_dur_amp : str or array
        Path to OpenFMRI stimulus file or 2D array containing three columns
        corresponding to onset, duration, amplitude.

    Returns
    -------
    block_spec : array
        Structured array with fields "start" (corresponding to onset time),
        "end" (onset time plus duration), "amplitude".
    """
    if not isinstance(ons_dur_amp, np.ndarray):
        ons_dur_amp = np.loadtxt(ons_dur_amp)
    onsets, durations, amplitudes = ons_dur_amp.T
    block_spec = make_recarray(
        np.column_stack((onsets, onsets + durations, amplitudes)),
        names=['start', 'end', 'amplitude'])
    # 2D arrays created by default by recarray when pased array input. This is
    # probably a bug, but work round here.
    return np.squeeze(block_spec)
