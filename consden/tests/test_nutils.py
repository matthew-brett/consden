""" Testing nutils module
"""

import numpy as np

from ..nutils import spline_basis, delta_basis, step_basis, t1_basis

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


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
