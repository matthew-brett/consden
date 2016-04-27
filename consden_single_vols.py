from __future__ import print_function

import os
from os.path import expanduser, split as psplit, join as pjoin

import numpy as np

import nibabel as nib

from ds114 import ROOT_PATH
func_prefix = 'waf'

def ffn(fname):
    path, fpart = psplit(fname)
    return pjoin(path, func_prefix + fpart)


for dirpath, dirnames, filenames in os.walk(ROOT_PATH):
    print('In', dirpath)
    b_n_fpart = ffn('b_n.nii')
    if not b_n_fpart in filenames:
        continue
    n_img = nib.load(pjoin(dirpath, b_n_fpart))
    e_img = nib.load(pjoin(dirpath, ffn('b_e.nii')))
    n_0 = n_img.get_data()[..., 0]
    e_0 = e_img.get_data()[..., 0]
    diff_vol = n_0 - e_0
    nib.save(nib.Nifti1Image(n_0, n_img.affine, n_img.header),
             pjoin(dirpath, ffn('n_0.nii')))
    nib.save(nib.Nifti1Image(e_0, n_img.affine, n_img.header),
             pjoin(dirpath, ffn('e_0.nii')))
    nib.save(nib.Nifti1Image(diff_vol, n_img.affine, n_img.header),
             pjoin(dirpath, ffn('task_diff.nii')))
    break
