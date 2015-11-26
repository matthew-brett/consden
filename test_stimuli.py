""" Test stimuli module

Run tests with::

    nosetests test_stimuli.py
"""

import numpy as np
import numpy.testing as npt

import stimuli


def test_events2neural():
    # test events2neural function
    neural = stimuli.events2neural('cond_test1.txt', 2, 16)
    # cond_test1.txt file is:
    """
    10    5.0    1
    20    4.0    2
    24    3.0    0.1
    """
    # Expected values for tr=2, n_trs=16
    expected = np.zeros(16)
    expected[5:7] = 1
    expected[10:12] = 2
    expected[12] = 0.1
    npt.assert_array_equal(neural, expected)
    neural = stimuli.events2neural('cond_test1.txt', 1, 30)
    # Expected values for tr=1, n_trs=30
    expected = np.zeros(30)
    expected[10:15] = 1
    expected[20:24] = 2
    expected[24:27] = 0.1
    npt.assert_array_equal(neural, expected)
