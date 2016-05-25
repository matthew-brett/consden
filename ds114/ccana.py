""" Check correlation between fixed and unfixed parameter estimates

"fixed" refers to the voxel parameter estimates in a constrained model that
includes the noise regressors.

"unfixed" refers to the OLS voxel parameter estimates for the model without the
noise regressors.
"""
from __future__ import print_function

from os.path import split as psplit, join as pjoin

import numpy as np

import nibabel as nib

from . import gen_models


def get_sub_models(subjects):
    """ Return a dictionary of models keyed by subject, run no

    Parameters
    ----------
    subjects : sequence
        Subject numbers to collect data for.

    Returns
    -------
    sub_models : dict
        Dictionary with keys giving subject numbers.  Each value is a dict with
        keys giving the run number.  The values for these keys are dicts with
        keys "fixed", "unfixed", "mask" containing the fixed parameter volume,
        the unfixed parameter volume, and the mask volume for the corresponding
        subject and run.
    """
    sub_models = {}
    all_sub_mask = None
    for model in gen_models(subjects):
        this_run = model.run
        sub_no = this_run.subject.subj_no
        if sub_no not in sub_models:
            sub_models[sub_no] = {}
        run_no = model.run_no
        func_fname = this_run.get_bold_fname()
        out_path, fpart = psplit(func_fname)
        unfixed = pjoin(out_path, 'wafn_0.nii')
        fixed = pjoin(out_path, 'wafe_0.nii')
        mask = pjoin(out_path, 'wafmask.nii')
        mask_data = nib.load(mask).get_data() != 0
        if all_sub_mask is None:
            all_sub_mask = mask_data
        else:
            all_sub_mask = all_sub_mask & mask_data
        data = dict(unfixed=nib.load(unfixed).get_data(),
                    fixed=nib.load(fixed).get_data(),
                    mask=mask_data)
        sub_models[sub_no][run_no] = data
    return sub_models, all_sub_mask


def img_cc(img1, img2):
    """ Pearson PM correlation between two images

    Equivalent to, but faster than::

        np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
    """
    x_dash = img1.ravel()
    x_dash = x_dash - x_dash.mean()
    y_dash = img2.ravel()
    y_dash = y_dash - y_dash.mean()
    return (x_dash.dot(y_dash) /
            np.sqrt(x_dash.dot(x_dash)) /
            np.sqrt(y_dash.dot(y_dash)))


def models_to_blocks(sub_models):
    """ Return 5D array of parameter volumes collated from subject models

    Parameters
    ----------
    sub_models : dict
        Dictionary with keys giving subject numbers.  Each value is a dict with
        a single key, the run number.  The value for this key is a dict with
        keys "fixed", "unfixed", "mask" containing the fixed parameter volume,
        the unfixed parameter volume, and the mask volume for the corresponding
        subject and run

    Returns
    -------
    fixed_block : shape (S, R, I, J K) array
        5D array containing fixed (denoised) parameter estimates for each
        subject and run.  S is the number of subject, R is the number of runs,
        (I, J, K) is the shape of one parameter volume.
    unfixed_block : shape (S, R, I, J K) array
        Corresponding 5D array for unfixed (OLS) parameter estimates.
    """
    n_subs = len(sub_models)
    unfixed_block = None
    for sub_i, sub_no in enumerate(sorted(sub_models)):
        for run_i, run_no in enumerate(sub_models[sub_no]):
            data = sub_models[sub_no][run_no]
            if unfixed_block is None:
                vol_shape = data['unfixed'].shape
                unfixed_block = np.zeros((n_subs, 2) + vol_shape)
                fixed_block = np.zeros((n_subs, 2) + vol_shape)
            unfixed_block[sub_i, run_i] = data['unfixed']
            fixed_block[sub_i, run_i] = data['fixed']
    return fixed_block, unfixed_block


def intra_extra_corr(sub_models):
    """ Correlation across runs for fixed, unfixed parameter volumes

    Parameters
    ----------
    sub_models : dict
        Dictionary with keys giving subject numbers.  Each value is a dict with
        keys giving the run number.  The values for these keys are dicts with
        keys "fixed", "unfixed", "mask" containing the fixed parameter volume,
        the unfixed parameter volume, and the mask volume for the corresponding
        subject and run.

    Returns
    -------
    intras : array shape (S, R)
        S is the number of subjects, R is the number of runs (=2).  ``intras[:,
        0]`` is the correlation of the "unfixed" parameters volumes between run
        1 and run 2, ``intras[:, 1]`` is the correlation between the "fixed"
        volumes.  Correlations calculated only for voxels within the "mask" for
        both runs.
    extras : array shape (S, R)
        ``extras[:, 0]`` is the correlation of the "unfixed" parameter volume
        for run 1 and the "fixed" volume for run 2.  ``extras[:, 1]`` is the
        correlation between "fixed" for run 1 against "unfixed" for run 2.
        Correlations masked as for `intras`.
    """
    n_subs = len(sub_models)
    intras = np.zeros((n_subs, 2))
    extras = np.zeros((n_subs, 2))

    for sub_i, sub_no in enumerate(sorted(sub_models)):
        unfixeds = []
        fixeds = []
        masks = []
        for run_i, run_no in enumerate(sub_models[sub_no]):
            data = sub_models[sub_no][run_no]
            unfixeds.append(data['unfixed'])
            fixeds.append(data['fixed'])
            masks.append(data['mask'])
        mask = masks[0] & masks[1]
        unfixeds = [n[mask] for n in unfixeds]
        fixeds = [e[mask] for e in fixeds]
        intras[sub_i, 0] = img_cc(*unfixeds)
        intras[sub_i, 1] = img_cc(*fixeds)
        extras[sub_i, 0] = img_cc(unfixeds[0], fixeds[1])
        extras[sub_i, 1] = img_cc(fixeds[0], unfixeds[1])
    return intras, extras


def leave_one_out(fixed_block, unfixed_block):
    """ Leave out one subject for group unfixed vs left-out fixed / unfixed

    For each subject, calculate mean of every subject but this one, across all
    subjects and runs.  Calculate mean across runs for left-out subject, for
    "fixed" and "unfixed". Correlate mean across subjects against mean over
    runs for left-out subject.

    Parameters
    ----------
    fixed_block : shape (S, R, I, J K) array
        5D array containing fixed (denoised) parameter estimates for each
        subject and run.  S is the number of subject, R is the number of runs,
        (I, J, K) is the shape of one parameter volume.
    unfixed_block : shape (S, R, I, J K) array
        Corresponding 5D array for unfixed (OLS) parameter estimates.

    Returns
    -------
    leave_one_outs : array shape (S, R)
        Values in first column $f_0, f_1 ... f_i ... f_{S-1}$ give correlation
        between [mean of "unfixed" across all subjects, runs except those for
        subject $i$] and [mean of "fixed" across runs for subject $i$].
        Second column gives correlations between [mean of "unfixed" across all
        subjects, runs except those for subject $i$] and [mean of "unfixed"
        across runs for subject $i$].
    """
    n_subs = fixed_block.shape[0]
    leave_one_outs = np.zeros((n_subs, 2))
    all_subs = np.arange(n_subs)
    vol_shape = fixed_block.shape[2:]
    for sub_i in range(n_subs):
        other_blocks = unfixed_block[all_subs != sub_i]
        other_mean = other_blocks.reshape((-1,) + vol_shape).mean(axis=0)
        these_fixed = fixed_block[sub_i].mean(axis=0)
        these_unfixed = unfixed_block[sub_i].mean(axis=0)
        leave_one_outs[sub_i, 0] = img_cc(other_mean, these_fixed)
        leave_one_outs[sub_i, 1] = img_cc(other_mean, these_unfixed)
    return leave_one_outs
