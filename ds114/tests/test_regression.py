""" Test regressions against original results

Of course, the originals may well be wrong, so this is to raise a warning that
something has changed, in case you weren't expecting that.
"""

from os.path import dirname, join as pjoin

from random import randint

import numpy as np

import nibabel as nib

from ds114 import SUBJECTS, gen_models
from ds114.ccana import (get_sub_models, models_to_blocks, intra_extra_corr,
                         leave_one_out)

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


# Columns in this array reversed compared to current code
ORIGINAL_L1OUTS = np.array([[ 0.26876784,  0.2684226 ],
                            [ 0.2139732 ,  0.21473651],
                            [ 0.47378051,  0.4731655 ],
                            [ 0.44217325,  0.44284857],
                            [ 0.10986545,  0.10952894],
                            [ 0.30505719,  0.30544015],
                            [ 0.46049827,  0.46057603],
                            [ 0.35226477,  0.35083325],
                            [ 0.42747262,  0.42710073],
                            [ 0.40689918,  0.40649565]])


ORIGINAL_INTRAS = np.array([[ 0.19271507,  0.19022251],
                            [ 0.50873593,  0.51076636],
                            [ 0.57513808,  0.57387731],
                            [ 0.59025563,  0.5938239 ],
                            [ 0.4568975 ,  0.45783393],
                            [ 0.36193661,  0.36540531],
                            [ 0.45300493,  0.45230072],
                            [ 0.16510891,  0.16506418],
                            [ 0.53811337,  0.53784655],
                            [ 0.57775963,  0.57641765]])

ORIGINAL_EXTRAS = np.array([[ 0.19193069,  0.19100509],
                            [ 0.51035792,  0.50912425],
                            [ 0.57380944,  0.57520478],
                            [ 0.5932119 ,  0.59087885],
                            [ 0.45933695,  0.45546101],
                            [ 0.36412965,  0.36323512],
                            [ 0.45354427,  0.45174901],
                            [ 0.16525359,  0.16491777],
                            [ 0.53830507,  0.53765976],
                            [ 0.57796617,  0.57621037]])


SUB_MODELS, ALL_SUB_MASK = get_sub_models(SUBJECTS)
FIXED_BLOCK, UNFIXED_BLOCK = models_to_blocks(SUB_MODELS)
# Make masks same for all subjects / runs
UNFIXED_BLOCK[:, :, ~ALL_SUB_MASK] = 0
FIXED_BLOCK[:, :, ~ALL_SUB_MASK] = 0


def test_subjects_regression():
    assert_array_equal([s.subj_no for s in SUBJECTS], range(1, 11))


def test_sub_models():
    models = list(gen_models(SUBJECTS))
    model = models[randint(0, len(models)-1)]
    run = model.run
    data = SUB_MODELS[run.subject.subj_no][run.run_no]
    out_path = dirname(run.get_bold_fname())
    data_for = lambda f: nib.load(pjoin(out_path, f)).get_data()
    assert_array_equal(data['unfixed'], data_for('wafn_0.nii'))
    assert_array_equal(data['fixed'], data_for('wafe_0.nii'))
    assert_array_equal(data['mask'], data_for('wafmask.nii') != 0)


def test_intras_extras_regression():
    intras, extras = intra_extra_corr(SUB_MODELS)
    assert_almost_equal(intras, ORIGINAL_INTRAS)
    assert_almost_equal(extras, ORIGINAL_EXTRAS)


def test_intras_extras():
    intras, extras = intra_extra_corr(SUB_MODELS)
    s3 = SUB_MODELS[3]
    r1, r2 = s3[1], s3[2]
    mask = r1['mask'] & r2['mask']
    r1f = r1['fixed'][mask]
    r1u = r1['unfixed'][mask]
    r2f= r2['fixed'][mask]
    r2u = r2['unfixed'][mask]
    cc = lambda x, y: np.corrcoef(x, y)[0, 1]
    assert_almost_equal(intras[2, 0], cc(r1u, r2u))
    assert_almost_equal(intras[2, 1], cc(r1f, r2f))
    assert_almost_equal(extras[2, 0], cc(r1u, r2f))
    assert_almost_equal(extras[2, 1], cc(r1f, r2u))


def test_leave_one_outs():
    leave_one_outs = leave_one_out(FIXED_BLOCK, UNFIXED_BLOCK)
    assert_almost_equal(leave_one_outs, np.fliplr(ORIGINAL_L1OUTS))
