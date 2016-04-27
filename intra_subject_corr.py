""" Check correlation between fixed and unfixed
"""
from __future__ import print_function

from os.path import (expanduser, split as psplit, join as pjoin, exists)

import numpy as np

import nibabel as nib

from ds114 import SUBJECTS, gen_models

sub_models = {}
all_sub_mask = None

S = len(SUBJECTS)

for model in gen_models(SUBJECTS):
    this_run = model.run
    sub_no = this_run.subject.subj_no
    if sub_no not in sub_models:
        sub_models[sub_no] = {}
    run_no = model.run_no
    func_fname = this_run.get_bold_fname()
    out_path, fpart = psplit(func_fname)
    naive = pjoin(out_path, 'wafn_0.nii')
    exper = pjoin(out_path, 'wafe_0.nii')
    mask = pjoin(out_path, 'wafmask.nii')
    mask_data = nib.load(mask).get_data() != 0
    if all_sub_mask is None:
        all_sub_mask = mask_data
    else:
        all_sub_mask = all_sub_mask & mask_data
    data = dict(naive=nib.load(naive).get_data(),
                exper=nib.load(exper).get_data(),
                mask=mask_data)
    sub_models[sub_no][run_no] = data


def img_cc(img1, img2):
    return np.corrcoef(img1.ravel(), img2.ravel())[0, 1]


# Do intra-analysis and extra-analysis correlations
intras = np.zeros((S, 2))
extras = np.zeros((S, 2))

# Collect data into 5D block
vol_shape = all_sub_mask.shape
naive_block = np.zeros((S, 2) + vol_shape)
exper_block = np.zeros((S, 2) + vol_shape)

# Do intra-analysis and cross-analysis correlation
for sub_i, sub_no in enumerate(sorted(sub_models)):
    naives = []
    expers = []
    masks = []
    for run_i, run_no in enumerate(sub_models[sub_no]):
        data = sub_models[sub_no][run_no]
        naives.append(data['naive'])
        expers.append(data['exper'])
        masks.append(data['mask'])
        naive_block[sub_i, run_i] = data['naive']
        exper_block[sub_i, run_i] = data['exper']
    mask = masks[0] & masks[1]
    naives = [n[mask] for n in naives]
    expers = [e[mask] for e in expers]
    intras[sub_i, 0] = img_cc(*naives)
    intras[sub_i, 1] = img_cc(*expers)
    extras[sub_i, 0] = img_cc(naives[0], expers[1])
    extras[sub_i, 1] = img_cc(expers[0], naives[1])

naive_block[:, :, ~all_sub_mask] = 0
exper_block[:, :, ~all_sub_mask] = 0


# For each subject, calculate mean of every subject but this one.
# Correlate with mean of two exper runs and naive runs for this subject.
leave_one_outs = np.zeros((S, 2))
all_subs = np.arange(S)
for sub_i in range(S):
    other_blocks = naive_block[all_subs != sub_i]
    other_mean = other_blocks.reshape((-1,) + vol_shape).mean(axis=0)
    these_exper = exper_block[sub_i].mean(axis=0)
    these_naive = naive_block[sub_i].mean(axis=0)
    leave_one_outs[sub_i, 0] = img_cc(other_mean, these_naive)
    leave_one_outs[sub_i, 1] = img_cc(other_mean, these_exper)
