from __future__ import print_function

from os.path import expanduser, split as psplit, join as pjoin, exists

import numpy as np

import nibabel as nib

from consden.nutils import openfmri2nipy, T1_GRAY_1p5T
from consden.designs import analyze_4d
from consden.openfmri import get_subjects

root_path = expanduser('~/data/ds114')
subjects = get_subjects(root_path)
model_no = 0
t1_constant = T1_GRAY_1p5T
n_dummies = 4
dct_order = 8
func_prefix = 'waf'
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

for subject in subjects:
    model = subject.models[0]
    for run_model in model.run_models:
        if run_model.task_no != 2:
            continue
        func_fname = run_model.run.get_bold_fname()
        func_fname = or_gz(prefix_fname(func_fname, func_prefix))
        print('Processing', func_fname)
        cond_no, ons_dur_amp = run_model.conditions[0]
        block_spec = openfmri2nipy(ons_dur_amp)
        contrasts, B_n, B_e, B_c, mask  = analyze_4d([('motor', block_spec)],
                                                     func_fname,
                                                     t1_constant,
                                                     TR = TR,
                                                     n_dummies=n_dummies,
                                                     dct_order=dct_order)
        out_path, _ = psplit(func_fname)
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
