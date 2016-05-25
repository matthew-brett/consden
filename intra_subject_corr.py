""" Check correlation between fixed and unfixed parameter estimates

"fixed" refers to the voxel parameter estimates in a constrained model that
includes the noise regressors.

"unfixed" refers to the OLS voxel parameter estimates for the model without the
noise regressors.
"""
from __future__ import print_function

from ds114 import SUBJECTS
from ds114.ccana import (get_sub_models, models_to_blocks, intra_extra_corr,
                         leave_one_out)

sub_models, all_sub_mask = get_sub_models(SUBJECTS)
fixed_block, unfixed_block = models_to_blocks(sub_models)

# Correlation across runs for fixed, unfixed (intras)
# Correlation across runs fixed -> unfixed, unfixed -> fixed
intras, extras = intra_extra_corr(sub_models)

# Make masks same for all subjects / runs
unfixed_block[:, :, ~all_sub_mask] = 0
fixed_block[:, :, ~all_sub_mask] = 0

# Leave one out correlations. Correlations with unfixed mean of all subjects
# excluding one.  First column is correlation with unfixed volumes from left
# out subject, second column is correlation with fixed volumes from left out
# subject.
leave_one_outs = leave_one_out(fixed_block, unfixed_block)
