from __future__ import print_function

from os.path import (expanduser, split as psplit, join as pjoin, exists)

MULTI = True

if MULTI:
    import multiprocessing

import numpy as np

import nibabel as nib
from nibabel.filename_parser import splitext_addext

from consden.nutils import openfmri2nipy, T1_GRAY_1p5T
from consden.designs import analyze_4d, get_vol_times, compile_design
from consden.openfmri import get_subjects

root_path = expanduser('~/data/ds114')
subjects = get_subjects(root_path)
model_no = 0
t1_constant = T1_GRAY_1p5T
n_dummies = 4  # Number of dummy scans where first dummy starts at T=0
n_removed = 4  # Number of dummy scans already removed from 4D functional
dct_order = 8
func_prefix = 'waf'
func_rp_prefix = 'af'
TR = 2.5


def prefix_fname(fname, prefix):
    path, name = psplit(fname)
    return pjoin(path, prefix + name)


def or_gz(fname):
    if fname.endswith('.gz'):
        shorter = fname[:-3]
        if exists(shorter):
            return shorter
    return fname


def analyze_model(run_model):
    func_fname = run_model.run.get_bold_fname()
    out_path, fpart = psplit(func_fname)
    # Get motion parameters
    froot = splitext_addext(fpart)[0]
    rp_fname = pjoin(out_path, 'rp_' + func_rp_prefix + froot + '.txt')
    motion_regressors = np.loadtxt(rp_fname)
    func_fname = or_gz(prefix_fname(func_fname, func_prefix))
    # Acquisition times of each volume that will remain
    vol_times = get_vol_times(func_fname,
                              n_dummies=n_dummies,
                              n_removed=n_removed,
                              TR=TR)
    cond_no, ons_dur_amp = run_model.conditions[0]
    block_spec = openfmri2nipy(ons_dur_amp)
    X, cons = compile_design(vol_times,
                             [('motor', block_spec)],
                              extra_cols=motion_regressors,
                              dct_order=dct_order)
    print('Processing', func_fname)
    # B_n is naive fit without noise model
    # B_e is fit for experimental part of design when modeling noise
    # B_c is fit for noise
    B_n, B_e, B_c, mask = analyze_4d(vol_times,
                                     X,
                                     func_fname,
                                     t1_constant,
                                     n_dummies=n_dummies - n_removed)
    nib.save(B_n, pjoin(out_path, func_prefix + 'b_n.nii'))
    nib.save(B_e, pjoin(out_path, func_prefix + 'b_e.nii'))
    nib.save(B_c, pjoin(out_path, func_prefix + 'b_c.nii'))
    nib.save(mask, pjoin(out_path, func_prefix + 'mask.nii'))
    task_diff = B_n.get_data()[..., 0] - B_e.get_data()[..., 0]
    task_diff = nib.Nifti1Image(task_diff, B_n.affine, B_n.header)
    nib.save(task_diff, pjoin(out_path, func_prefix + 'task_diff.nii'))
    msk = mask.get_data().astype(bool)
    coefs = B_c.get_data()[msk].T
    print(np.where(np.sum(coefs, axis=1))[0])


def gen_models(subjects):
    for subject in subjects:
        model = subject.models[0]
        for run_model in model.run_models:
            if run_model.task_no != 2:
                continue
            yield run_model


if __name__ == '__main__':
    jobs = []
    for model in gen_models(subjects):
        if MULTI:
            p = multiprocessing.Process(target=analyze_model, args=(model,))
            jobs.append(p)
            p.start()
        else:  # Serial run
            analyze_model(model)
