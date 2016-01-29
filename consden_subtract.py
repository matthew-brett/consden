from __future__ import print_function

import os
from os.path import expanduser, split as psplit, join as pjoin

import numpy as np

import nibabel as nib

root_path = expanduser('~/data/ds114')

for dirpath, dirnames, filenames in os.walk(root_path):
    if not 'b_n.nii' in filenames:
        continue
    n_img = nib.load(pjoin(dirpath, 'b_n.nii'))
    e_img = nib.load(pjoin(dirpath, 'b_e.nii'))
    diff_vol = n_img.get_data()[..., 0] - e_img.get_data()[..., 0]
    nib.save(nib.Nifti1Image(diff_vol, n_img.affine, n_img.header),
             pjoin(dirpath, 'task_diff.nii'))
    mask = diff_vol == 0
    c_img = nib.load(pjoin(dirpath, 'b_c.nii'))
    coeffs = c_img.get_data()[mask].T.astype(float)
    nz = np.sum(coeffs, axis=0)
    print(dirpath, nz, mask.shape, c_img.shape)
