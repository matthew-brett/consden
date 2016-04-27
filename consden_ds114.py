from __future__ import print_function

from os.path import (split as psplit, join as pjoin, exists)

MULTI = True

if MULTI:
    import multiprocessing

import numpy as np

import nibabel as nib
from nibabel.filename_parser import splitext_addext

from consden.nutils import openfmri2nipy, T1_GRAY_1p5T
from consden.designs import analyze_4d, get_vol_times, compile_design

from ds114 import TR, SUBJECTS, gen_models


T1_CONSTANT = T1_GRAY_1p5T
# Number of frames to drop before the analysis.  If dummy scans already
# dropped, set this to 0, and set FRAME0_START_TIME to reflect frame start
# time, relative to the design.
N_TO_DROP = 0
# Start time of first frame included in the analysis.  This will likely not be
# 0 if N_TO_DROP is not 0, or previous analysis has dropped frames.
FRAME0_START_TIME = TR * 4
DCT_ORDER = 8

MODEL_NO=0
TASK_NOS=[2]

# Where to find files on the filesystem
FUNC_PREFIX = 'waf'
FUNC_RP_PREFIX = 'af'


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
    # Name of the image file of the functional run
    func_fname = run_model.run.get_bold_fname()
    out_path, fpart = psplit(func_fname)
    # Get motion parameters
    froot = splitext_addext(fpart)[0]
    rp_fname = pjoin(out_path, 'rp_' + FUNC_RP_PREFIX + froot + '.txt')
    motion_regressors = np.loadtxt(rp_fname)
    # If there's a file with the same name, without the .gz extension, use that
    # instead, assuming it's just the uncompressed version of the same file.
    func_fname = or_gz(prefix_fname(func_fname, FUNC_PREFIX))
    # Acquisition times of each volume that will remain, after we have (later)
    # dropped any scans at the beginning of the run.
    vol_times = get_vol_times(func_fname, TR)[N_TO_DROP:] + FRAME0_START_TIME
    cond_no, ons_dur_amp = run_model.conditions[0]
    block_spec = openfmri2nipy(ons_dur_amp)
    X, cons = compile_design(vol_times,
                             [('motor', block_spec)],
                              extra_cols=motion_regressors,
                              dct_order=DCT_ORDER)
    print('Processing', func_fname)
    # B_n is naive fit without noise model
    # B_e is fit for experimental part of design when modeling noise
    # B_c is fit for noise
    B_n, B_e, B_c, mask = analyze_4d(vol_times,
                                     X,
                                     func_fname,
                                     T1_CONSTANT,
                                     n_to_drop=N_TO_DROP)
    nib.save(B_n, pjoin(out_path, FUNC_PREFIX + 'b_n.nii'))
    nib.save(B_e, pjoin(out_path, FUNC_PREFIX + 'b_e.nii'))
    nib.save(B_c, pjoin(out_path, FUNC_PREFIX + 'b_c.nii'))
    nib.save(mask, pjoin(out_path, FUNC_PREFIX + 'mask.nii'))
    task_diff = B_n.get_data()[..., 0] - B_e.get_data()[..., 0]
    task_diff = nib.Nifti1Image(task_diff, B_n.affine, B_n.header)
    nib.save(task_diff, pjoin(out_path, FUNC_PREFIX + 'task_diff.nii'))
    msk = mask.get_data().astype(bool)
    coefs = B_c.get_data()[msk].T
    print(np.where(np.sum(coefs, axis=1))[0])


if __name__ == '__main__':
    jobs = []
    for model in gen_models(SUBJECTS, TASK_NOS, MODEL_NO):
        if MULTI:
            p = multiprocessing.Process(target=analyze_model, args=(model,))
            jobs.append(p)
            p.start()
        else:  # Serial run
            analyze_model(model)
