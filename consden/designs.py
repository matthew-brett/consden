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
    img = nib.Nifti1Image(data, img.affine, img.header)
    img.set_data_dtype(something_by_vox.dtype.type)
    return img


def get_vol_times(img, n_dummies=0, TR=None):
    """ Return volume onset times in seconds for image `img`

    Parameters
    ----------
    img : str or image object
        String giving image filename or nibabel image object with attributes
        ``shape`` and ``header``.
    n_dummies : int, optional
        Number of dummy scans in volume (frames at beginning of run which we
        will discard).
    TR : None or float, optional
        If None, try and get TR from `img`.  If TR in `img` seems to be an
        uninformative default raise ValueError.  If TR is not None, gives TR
        value in seconds.

    Returns
    -------
    vol_times : 1D array
        1D array of volume onset times in seconds, length ``img.shape[-1] -
        n_dummies``.
    """
    img = img if hasattr(img, 'shape') else nib.load(img)
    if TR is None:
        TR = img.header['pixdim'][4]
        if TR in (0, 1):
            raise ValueError("TR not valid in image, set with kwarg")
    return np.arange(n_dummies, img.shape[-1]) * TR


def compile_design(vol_times, block_infos, extra_cols=None, dct_order=8):
    """ Create design with given task specification and confounds

    Parameters
    ----------
    vol_times : 1D array
        Volume onset times in seconds.
    block_infos : sequence
        Sequence of length 2 sequences where elements are (name, blk_spec),
        where ``name`` is a string, and ``blk_spec`` is a recarray matching the
        format required for :func:`block_amplitudes`.
    extra_cols : None or 2D array, optional
        Extra columns to add to the design.
    dct_order : int, optional
        Order of DCT-II basis functions to append.

    Returns
    -------
    design : 2D array
        Design matrix
    contrasts : dict
        Dictionary of contrasts, one for each condition specified in
        `block_infos`.
    """
    exp_cons = [block_amplitudes(name, spec, vol_times)
                for name, spec in block_infos]
    if extra_cols is not None:
        exp_cons.append((extra_cols,))
    exp_cons.append((dct_ii_basis(vol_times, dct_order),))
    return stack_designs(*exp_cons)


def analyze_4d(vol_times, exp_design, bold_fname, t1_constant, n_dummies=0):
    img = nib.load(bold_fname)
    data = img.get_data()[..., n_dummies:]
    mean_data = np.mean(data, axis=-1)
    mask = compute_mask(mean_data)
    Y = data[mask].T
    assert Y.shape[0] == data.shape[-1]
    X_c = build_confounds(exp_design, vol_times, t1_constant)
    B_e, B_c = solve_with_confounds(Y, exp_design, X_c)
    B_e_naive = npl.pinv(exp_design).dot(Y)
    return ([_to_img(b, img, mask) for b in (B_e_naive, B_e, B_c)] +
            [nib.Nifti1Image(mask, img.affine, img.header)])


from nipy.modalities.fmri.hrf import glover
from nipy.modalities.fmri.utils import (T, convolve_functions, blocks)
from nipy.algorithms.statistics.formula.formulae import (
    Formula, make_recarray)


def block_amplitudes(name, block_spec, t, hrfs=(glover,),
                     convolution_padding=5.,
                     convolution_dt=0.02,
                     hrf_interval=(0.,30.)):
    """ Design matrix at times `t` for blocks specification `block_spec`

    Create design matrix for linear model from a block specification
    `block_spec`,  evaluating design rows at a sequence of time values `t`.

    `block_spec` may specify amplitude of response for each event, if different
    (see description of `block_spec` parameter below).

    The on-off step function implied by `block_spec` will be convolved with
    each HRF in `hrfs` to form a design matrix shape ``(len(t), len(hrfs))``.

    Parameters
    ----------
    name : str
        Name of condition
    block_spec : np.recarray or array-like
       A recarray having fields ``start, end, amplitude``, or a 2D ndarray /
       array-like with three columns corresponding to start, end, amplitude.
    t : np.ndarray
       An array of np.float values at which to evaluate the design. Common
       examples would be the acquisition times of an fMRI image.
    hrfs : sequence, optional
       A sequence of (symbolic) HRFs that will be convolved with each block.
       Default is ``(glover,)``.
    convolution_padding : float, optional
       A padding for the convolution with the HRF. The intervals
       used for the convolution are the smallest 'start' minus this
       padding to the largest 'end' plus this padding.
    convolution_dt : float, optional
       Time step for high-resolution time course for use in convolving the
       blocks with each HRF.
    hrf_interval: length 2 sequence of floats, optional
       Interval over which the HRF is assumed supported, used in the
       convolution.

    Returns
    -------
    X : np.ndarray
       The design matrix with ``X.shape[0] == t.shape[0]``. The number of
       columns will be ``len(hrfs)``.
    contrasts : dict
       A contrast is generated for each HRF specified in `hrfs`.
    """
    block_spec = np.asarray(block_spec)
    if block_spec.dtype.names is not None:
        if not block_spec.dtype.names in (('start', 'end'),
                                          ('start', 'end', 'amplitude')):
            raise ValueError('expecting fields called "start", "end" and '
                             '(optionally) "amplitude"')
        block_spec = np.array(block_spec.tolist())
    block_times = block_spec[:, :2]
    amplitudes = block_spec[:, 2] if block_spec.shape[1] == 3 else None
    # Now construct the design in time space
    convolution_interval = (block_times.min() - convolution_padding,
                            block_times.max() + convolution_padding)
    B = blocks(block_times, amplitudes=amplitudes)
    t_terms = []
    c_t = {}
    n_hrfs = len(hrfs)
    for hrf_no in range(n_hrfs):
        t_terms.append(convolve_functions(B, hrfs[hrf_no](T),
                                          convolution_interval,
                                          hrf_interval,
                                          convolution_dt))
        contrast = np.zeros(n_hrfs)
        contrast[hrf_no] = 1
        c_t['{0}_{1:d}'.format(name, hrf_no)] = contrast
    t_formula = Formula(t_terms)
    tval = make_recarray(t, ['t'])
    X_t = t_formula.design(tval, return_float=True)
    return X_t, c_t
