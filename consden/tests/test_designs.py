""" Testing designs module
"""

import numpy as np

from ..nutils import step_basis, t1_basis
from ..designs import (build_confounds)

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_build_confounds():
    N = 100
    vol_times = np.arange(N) * 3.1 + 11.4
    t1_constant = 13
    exp_design = np.c_[np.arange(N), np.ones(N)]
    confounds = build_confounds(exp_design, vol_times, t1_constant)
    steps = step_basis(vol_times)
    t1s = t1_basis(vol_times, t1_constant)
    exp_confounds = np.c_[np.eye(N), steps[:, 2:-1], t1s[:, :-1]]
    exp_confounds -= np.mean(exp_confounds, axis=0)
    assert_array_equal(confounds, exp_confounds)
