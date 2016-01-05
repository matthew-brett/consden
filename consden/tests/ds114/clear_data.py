from __future__ import print_function

import os
from os.path import join as pjoin

import numpy as np

import nibabel as nib
nib.volumeutils.default_compresslevel = 9

TEMPLATE_SHAPE = (4, 5, 6, 2)

for dirpath, dirnames, filenames in os.walk('.'):
    for fname in filenames:
        if not fname.endswith('.nii.gz'):
            continue
        fpath = pjoin(dirpath, fname)
        img = nib.load(fpath)
        shape = TEMPLATE_SHAPE[:len(img.shape)]
        img = nib.Nifti1Image(np.zeros(shape), img.affine, img.header)
        print('Saving', fpath)
        nib.save(img, fpath)
    pruned = []
    for dirname in dirnames:
        if dirname in ('dwi', 'QA'):
            os.removedirs(pjoin(dirpath, dirname))
        else:
            pruned.append(dirname)
    dirnames[:] = pruned
