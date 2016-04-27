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


b_n_imgs = []
b_e_imgs = []
for dirpath, dirnames, filenames in os.walk(ROOT_PATH):
    b_n_fpart = ffn('b_n.nii')
    b_e_fpart = ffn('b_e.nii')
    if b_n_fpart not in filenames:
        continue
    assert b_e_fpart in filenames
    b_n_imgs.append(pjoin(dirpath, b_n_fpart))
    b_e_imgs.append(pjoin(dirpath, b_e_fpart))

b_ns = nib.funcs.concat_images(b_n_imgs)
mean_b_ns = np.mean(b_ns.get_data(), axis=-1)
mean_bns = nib.Nifti1Image(mean_b_ns, b_ns.affine, b_ns.header)
nib.save(mean_bns, func_prefix + 'mean_b_n.nii')
b_es = nib.funcs.concat_images(b_e_imgs)
mean_b_es = np.mean(b_es.get_data(), axis=-1)
mean_bns = nib.Nifti1Image(mean_b_es, b_es.affine, b_es.header)
nib.save(mean_bns, func_prefix + 'mean_b_e.nii')

