import numpy as np

from .nutils import (drop_colin, demean_cols, delta_basis, step_basis,
                     t1_basis)


def build_confounds(exp_design, vol_times, t1_constant):
    """ Compile scan deltas, step functions and T1 decay into confound design

    Parameters
    ----------
    exp_design : array shape (N, P)
        Experimental design containing task and any confounds that should be
        given unbiased estimates.
    vol_times : array shape (N,)
        Vector of times at which each row of `exp_design` (volume) was
        collected.
    t1_constant : float
        T1 decay constant for T1 decay regressors.  See :func:`nutils.t1_basis`
        for detail.

    Returns
    -------
    confounds : array shape (N, C)
        Array containing columns with confound regressors, consisting of
        mean-centered columns encoding: deltas for each scan; step function
        starting at each scan; T1 decay starting at each scan.  We remove
        colinear columns within the confounds, and columns colinear with any
        column in the `exp_design`.
    """
    deltas = drop_colin(demean_cols(
        delta_basis(vol_times)), exp_design)
    steps = drop_colin(demean_cols(
        step_basis(vol_times)), np.c_[exp_design, deltas])
    t1s = drop_colin(demean_cols(
        t1_basis(vol_times, t1_constant)), np.c_[exp_design, deltas, steps])
    return np.c_[deltas, steps, t1s]
