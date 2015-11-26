"""
Test spm_funcs module

Run the tests with::

    nosetests test_spm_funcs.py
"""
# Python 3 compatibility
from __future__ import print_function, division

import numpy as np
import numpy.testing as npt

import spm_funcs

def test_spm_globals():
    # Test calculation of SPM global
    # Values as calculated by us
    glob_vals = spm_funcs.get_spm_globals('three_vols.nii')
    # Values as calculated using SPM
    expected_values = np.loadtxt('global_signals.txt')
    # These two sets of values should be very similar
    npt.assert_array_almost_equal(glob_vals, expected_values)


def test_spm_hrf():
    # Test calculation of SPM HRF
    # For a list of time gaps (gaps between time samples)
    for dt_str in ('0.1', '1', '2.5'):
        # Convert string e.g. 0.1 to string e.g. '0p1'
        dt_p_for_point = dt_str.replace('.', 'p')
        # Load saved values from SPM for this time difference
        spm_fname = 'spm_hrf_' + dt_p_for_point + '.txt'
        spm_values = np.loadtxt(spm_fname)
        # Time difference as floating point value
        dt = float(dt_str)
        # Make times corresponding to samples from SPM
        times = np.arange(len(spm_values)) * dt
        # Evaluate our version of the function at these times
        our_values = spm_funcs.spm_hrf(times)
        # Our values and the values from SPM should be about the same
        npt.assert_almost_equal(spm_values, our_values)
