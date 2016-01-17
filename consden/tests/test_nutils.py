""" Testing nutils module
"""

from os.path import dirname, join as pjoin

import numpy as np

from ..nutils import (spline_basis, delta_basis, step_basis, t1_basis,
                      drop_colin, demean_cols, dct_ii_basis, openfmri2nipy)

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

HERE = dirname(__file__)


def test_spline_basis():
    vt = np.array([-1, 3.1, 7, 10, 21, 33, 40])
    k = vt.mean()
    for order in range(1, 5):
        drift = np.array([vt**i for i in range(order + 1)] +
                         [(vt-k)**order * (np.greater(vt, k))] ).T
        drift /= np.max(drift, axis=0)
        assert_almost_equal(spline_basis(vt, order), drift)


def test_delta_basis():
    assert_array_equal(delta_basis(np.arange(10)), np.eye(10))
    assert_array_equal(delta_basis(np.arange(12) * 2.5 + 4), np.eye(12))


def test_step_basis():
    volume_times = [-1, 3.1, 7, 10, 21]
    N = len(volume_times)
    steps = step_basis(volume_times)
    exp_steps = np.zeros((N, N))
    for i in range(len(volume_times)):
        exp_steps[i:, i] = 1
    assert_array_equal(steps, exp_steps)


def test_t1_basis():
    volume_times = [-1, 3.1, 7, 10, 21]
    for t1 in (1, 5, 10):
        basis = t1_basis(volume_times, t1)
        vts = np.array(volume_times)
        exp_basis = np.zeros(basis.shape)
        for i, t in enumerate(volume_times):
            col = np.e ** (-(vts - t) / t1)
            col[vts < t] = 0
            exp_basis[:, i] = col
        assert_almost_equal(basis, exp_basis)


def test_drop_colin():

    def normed_1(delta, n):
        # A normalized vector with 1 + delta as first element
        vec = np.zeros(n)
        vec[0] = 1 + delta
        vec[1] = -np.sqrt(2 * delta + delta ** 2)
        return vec

    for N in (5, 100):
        eyes = np.eye(N)
        assert_array_equal(drop_colin(eyes), eyes)
        # Dependent columns don't get dropped only columns colinear with other
        # columns in the design
        ones = np.ones((N, 1))
        eyes_1 = np.c_[eyes, ones]
        assert_array_equal(drop_colin(eyes_1), eyes_1)
        # Colinear columns get dropped
        eyes_eyes = np.c_[eyes, np.eye(N)[:, :2]]
        ee_copy = eyes_eyes.copy()
        dropped = drop_colin(eyes_eyes)
        assert_array_equal(dropped, eyes)
        # Check that dropping retuns array with different memory from original
        dropped[0, 0] = 99
        assert_array_equal(eyes_eyes, ee_copy)
        eyes_eyes_2 = np.c_[np.eye(N)[:, :2], eyes]
        assert_array_equal(drop_colin(eyes_eyes_2), eyes)
        # Test normalization
        eyes_eyes_3 = np.c_[eyes * 4, np.eye(N)[:, :2]]
        assert_array_equal(drop_colin(eyes_eyes_3), eyes * 4)
        # Negative colinear
        eyes_eyes_m1 = np.c_[eyes, np.eye(N)[:, :2] * -1]
        assert_array_equal(drop_colin(eyes_eyes_m1), eyes)
        # Can use list
        assert_array_equal(drop_colin(eyes_eyes.tolist()), eyes)
        # Tolerance
        eps = np.finfo(float).eps
        # Delta below threshold
        small_eyes = np.c_[eyes, normed_1(eps * (N - 1), N)]
        assert_array_equal(drop_colin(small_eyes), eyes)
        # Delta above threshold
        small_eyes = np.c_[eyes, normed_1(eps * (N + 1), N)]
        assert_array_equal(drop_colin(small_eyes), small_eyes)
        # Change threshold
        tol = eps * (N + 2)
        assert_array_equal(drop_colin(small_eyes, tol=tol), eyes)
        small_eyes = np.c_[eyes, normed_1(eps * (N + 3), N)]
        assert_array_equal(drop_colin(small_eyes, tol=tol), small_eyes)
        # Different reference design
        assert_array_equal(drop_colin(eyes_1, eyes), ones)
        assert_array_equal(drop_colin(eyes, eyes[:, :2]), eyes[:, 2:])
        # Can use list
        assert_array_equal(drop_colin(eyes_1, eyes.tolist()), ones)
        # Dropping zero-length columns
        eyes_with_0 = np.c_[eyes, ones, np.zeros(N)]
        assert_array_equal(drop_colin(eyes_with_0), eyes_1)
        eyes_with_00 = np.c_[np.zeros(N), eyes, ones, np.zeros(N)]
        assert_array_equal(drop_colin(eyes_with_00), eyes_1)


def test_demean_cols():
    # Test removing the mean from columns
    X = np.arange(24).reshape((6, 4))
    assert_almost_equal(demean_cols(X).mean(axis=0), 0)
    X = np.arange(24).reshape((6, 4)) - 100
    assert_almost_equal(demean_cols(X).mean(axis=0), 0)


def test_dct_ii_basis():
    # Test DCT-II basis
    for N in (5, 10, 100):
        spm_fname = pjoin(HERE, 'dct_{0}.txt'.format(N))
        spm_mtx = np.loadtxt(spm_fname)
        vol_times = np.arange(N) * 15. + 3.2
        our_dct = dct_ii_basis(vol_times)
        # Check dot products of columns
        sq_col_lengths = np.ones(N) * N / 2.
        sq_col_lengths[0] = N
        assert_almost_equal(our_dct.T.dot(our_dct),
                            np.diag(sq_col_lengths))
        col_lengths = np.sqrt(sq_col_lengths)
        assert_almost_equal(our_dct / col_lengths, spm_mtx)
        # Normalize length
        our_normed_dct = dct_ii_basis(vol_times, normcols=True)
        assert_almost_equal(our_normed_dct, spm_mtx)
        assert_almost_equal(our_normed_dct.T.dot(our_normed_dct), np.eye(N))
        for i in range(N):
            assert_almost_equal(dct_ii_basis(vol_times, i) / col_lengths[:i],
                                spm_mtx[:, :i])
            assert_almost_equal(dct_ii_basis(vol_times, i, True),
                                spm_mtx[:, :i])


def test_openfmri2nipy():
    # Test loading / processing OpenFMRI stimulus file
    stim_file = pjoin(HERE, 'cond_test1.txt')
    ons_dur_amp = np.loadtxt(stim_file)
    onsets, durations, amplitudes = ons_dur_amp.T
    for in_param in (stim_file, ons_dur_amp):
        res = openfmri2nipy(in_param)
        assert_equal(res.dtype.names, ('start', 'end', 'amplitude'))
        assert_array_equal(res['start'], onsets)
        assert_array_equal(res['end'], onsets + durations)
        assert_array_equal(res['amplitude'], amplitudes)
