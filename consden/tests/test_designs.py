""" Testing designs module
"""

from os.path import dirname, join as pjoin

import numpy as np

import nibabel as nib

from nipy.labs.mask import compute_mask
from nipy.modalities.fmri import design as fmrid

import regreg.api as rr

from ..nutils import step_basis, t1_basis, dct_ii_basis, openfmri2nipy
from ..designs import build_confounds, solve_with_confounds, analyze_4d

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


HERE = dirname(__file__)


def test_build_confounds():
    N = 100
    vol_times = np.arange(N) * 3.1 + 11.4
    t1_constant = 13
    exp_design = np.c_[np.arange(N), np.ones(N)]
    confounds = build_confounds(exp_design, vol_times, t1_constant)
    steps = step_basis(vol_times)
    t1s = t1_basis(vol_times, t1_constant)
    exp_confounds = np.c_[np.eye(N), steps[:, 2:-1], t1s[:, :-1]]
    exp_confounds -= np.mean(exp_confounds, axis=0)
    assert_array_equal(confounds, exp_confounds)


def test_big_utils():
    # Overall analysis
    func_fname = pjoin(HERE, 'small_func.nii')
    task_fname = pjoin(HERE, 'ds114_sub009_t2r1_cond.txt')
    img = nib.load(func_fname)
    TR = img.header['pixdim'][4]  # Not always reliable
    n_dummies = 4
    dct_order = 3
    t1_constant = 5.0
    data = img.get_data()
    data = data[..., n_dummies:]
    mean_data = data.mean(axis=-1)
    mask = compute_mask(mean_data)
    vox_by_time = data[mask]
    vol_times = np.arange(n_dummies, img.shape[-1]) * TR
    dct_basis = dct_ii_basis(vol_times, dct_order)
    block_spec = openfmri2nipy(task_fname)
    experiment, cons = fmrid.block_design(block_spec, vol_times)
    X_e = np.column_stack((experiment, dct_basis))
    X_c = build_confounds(X_e, vol_times, t1_constant)
    # Build regreg model by hand
    Y = vox_by_time.T
    design = X_e
    n_tr, n_vox = Y.shape
    n_basis = X_c.shape[1]
    design_pinv = np.linalg.pinv(design)
    # Projector onto space of experimental design
    P_design = np.dot(design, design_pinv)
    # Residual-forming matrix
    R_design = np.eye(n_tr) - P_design
    # Residuals of data w.r.t design
    resid_design = np.dot(R_design, Y)
    # X matrix for problem should be residualized w.r.t. design as well 
    X_ort_design = np.dot(R_design, X_c)
    loss_resid = rr.squared_error(X_ort_design, resid_design)
    loss_resid.shape = (n_basis, n_vox)
    penalty = rr.l1_l2(loss_resid.shape, lagrange=1.)
    dual_penalty = penalty.conjugate
    lam_max = dual_penalty.seminorm(
        loss_resid.smooth_objective(np.zeros(loss_resid.shape), 'grad'),
        lagrange=1)
    penalty.lagrange = 0.9 * lam_max
    problem_resid = rr.simple_problem(loss_resid, penalty)
    soln_resid = problem_resid.solve(min_its=200, tol=1.e-12)
    # Only one coefficient survives
    assert_equal(np.where((soln_resid**2).sum(1) != 0)[0], [2])
    # Reconstructing estimates for experimental design
    coefs_resid = np.dot(design_pinv, Y - np.dot(X_c, soln_resid))
    # Coefficients if estimating without sparse part of design
    coefs = design_pinv.dot(Y)
    # Now, test big-picture stuff
    # solve_with_confounds
    B_e, B_c = solve_with_confounds(Y, X_e, X_c)
    assert_array_equal(B_e, coefs_resid)
    assert_array_equal(B_c, soln_resid)
    # Make volume versions of coefficients
    task_resid = np.zeros((mean_data.shape + (coefs_resid.shape[0],)))
    task = task_resid.copy()
    task_resid[mask] = coefs_resid.T
    task[mask] = coefs.T
    confounds = np.zeros((mean_data.shape + (soln_resid.shape[0],)))
    confounds[mask] = soln_resid.T
    # analyze_4d
    contrasts, B_n, B_e, B_c, img_mask = analyze_4d([block_spec],
                                                    func_fname,
                                                    t1_constant,
                                                    n_dummies=n_dummies,
                                                    dct_order=dct_order)
    assert_array_equal(task, B_n.get_data())
    assert_array_equal(task_resid, B_e.get_data())
    assert_array_equal(confounds, B_c.get_data())
    assert_array_equal(mask, img_mask.get_data())
    for out_img in (B_n, B_e, B_c, img_mask):
        assert_array_equal(out_img.affine, img.affine)
        assert_array_equal(out_img.shape[:3], img.shape[:3])
    for beta_img in (B_n, B_e, B_c):
        assert_equal(beta_img.get_data_dtype().type, np.float64)
