""" Processing designs for OpenFMRI """

import numpy as np
import numpy.linalg as npl

from six import string_types

import nibabel as nib

import regreg.api as rr

from nipy.labs.mask import compute_mask
from nipy.modalities.fmri.design import block_design, stack_designs

from .nutils import (drop_colin, demean_cols, delta_basis, step_basis,
                     t1_basis, dct_ii_basis)


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


def solve_with_confounds(Y, X_exp, X_confounds):
    """ Solve hybrid design Y = [X_exp X_confounds][B_exp; B_confounds] + E

    where we also minimize the l1, l2 norm of B_confounds.

    Parameters
    ----------
    Y : array shape (N, S)
        The data
    X_exp : array shape (N, P)
        The experimental design.  We apply no penalty to the (P, S) parameters
        for `X_exp`.
    X_confounds : array shape (N, C)
        The confounds.  We minimize according to the l1,l2 norm of the
        parameters for `X_confounds`.

    Returns
    -------
    B_exp : array shape (P, S)
        Parameters for `X_exp`.
    B_confounds : array shape (C, S)
        Parameters for `X_confounds`.
    """
    N = X_exp.shape[0]
    C = X_confounds.shape[-1]
    S = Y.shape[-1]
    X_exp_pinv = npl.pinv(X_exp)
    # Projector onto space of experimental design
    P_exp = np.dot(X_exp, X_exp_pinv)
    # Residual-forming matrix
    R_exp = np.eye(N) - P_exp
    # Residuals of data w.r.t design
    Y_exp_resid = np.dot(R_exp, Y)
    # X matrix for confounds should be residualized w.r.t. design as well.
    confounds_exp_resid = np.dot(R_exp, X_confounds)
    loss_resid = rr.squared_error(confounds_exp_resid, Y_exp_resid)
    loss_resid.shape = (C, S)
    penalty = rr.l1_l2(loss_resid.shape, lagrange=1.)
    dual_penalty = penalty.conjugate
    lam_max = dual_penalty.seminorm(
        loss_resid.smooth_objective(np.zeros(loss_resid.shape), 'grad'),
        lagrange=1)
    penalty.lagrange = 0.9 * lam_max
    problem_resid = rr.simple_problem(loss_resid, penalty)
    # Parameters for residualized confounds applying l1, l2 norm
    B_confounds = problem_resid.solve(min_its=200, tol=1.e-12)
    # Reconstructing estimates for experimental design
    B_exp_design = np.dot(X_exp_pinv, Y - np.dot(X_confounds, B_confounds))
    return B_exp_design, B_confounds


def _to_3d(something_by_vox, vol_shape, mask, fill=0):
    """ Convert N by V array to 4D shape `vol_shape` + (N,) using `mask`
    """
    T = something_by_vox.shape[0]
    out = np.zeros(vol_shape + (T,))
    if fill != 0:
        out += fill
    out[mask] = something_by_vox.T
    return out


def _to_img(something_by_vox, img, mask, fill=0):
    vol_shape = img.shape[:-1]
    data = _to_3d(something_by_vox, vol_shape, mask, fill)
    return nib.Nifti1Image(data, img.affine, img.header)


def analyze_4d(block_specs, bold_fname, t1_constant, n_dummies=0, TR=None,
              dct_order=8):
    img = nib.load(bold_fname)
    data = img.get_data()[..., n_dummies:]
    if TR is None:
        TR = img.header['pixdim'][4]
        if TR in (0, 1):
            raise ValueError("TR not valid in image, set with kwarg")
    vol_times = np.arange(n_dummies, img.shape[-1]) * TR
    mean_data = np.mean(data, axis=-1)
    mask = compute_mask(mean_data)
    Y = data[mask].T
    assert Y.shape[0] == data.shape[-1]
    exp_cons = [block_design(spec, vol_times) for spec in block_specs]
    exp_cons.append((dct_ii_basis(vol_times, dct_order),))
    X_e, contrasts = stack_designs(*exp_cons)
    X_c = build_confounds(X_e, vol_times, t1_constant)
    B_e, B_c = solve_with_confounds(Y, X_e, X_c)
    B_e_naive = npl.pinv(X_e).dot(Y)
    return [contrasts] + [_to_img(b, img, mask)
                          for b in (B_e_naive, B_e, B_c)]
