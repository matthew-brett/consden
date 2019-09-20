######################
Constrained regression
######################

**********
Background
**********

Here we are trying constrained regression to detect outlier volumes in FMRI
time series.

For an introduction to the basic problem of outlier detection, please see:

"Methods to detect, characterize, and remove motion artifact in resting
state fMRI" Jonathan D. Power (2014) Neuroimage 84: 320-341.

For general background on constrained regression, please see the `RegReg
site <https://regreg.github.io/regreg>`_.

For the specific programme here, to use constrained regression for outlier
detection, see the "Research" section in `this talk
<https://bitbucket.org/matthewbrett/birmingham-letter/raw/9743d44d6f5bd8ea42102349c274d4ae09022eff/bham_slides.pdf>`_

In short, we are going to do the following:

* Set up a design matrix ``X_s`` ('s' for "standard") to express the design of
  the experiment, plus some usual experimental confounds;
* Make a noise design with Dirac delta and set-function regressors expressing the
  effect of movement at that time point.  Call the matrix containing these
  columns ``X_n`` ('n' for "noise).
* Call the column concatenation of these matrices ``X_f`` ('f' for "full"). The
  design is now heavily redundant, with many more columns than rows.
* Use constrained regression to estimate parameters for this design, where we
  use ordinary least squares for the ``X_s`` part of the design, and least
  squares plus constraints on the whole scans (rows) by noise parameters ``X_n``
  part of the design, in order to force parameters to zero unless there is
  strong evidence for effect across the whole row.
* Our intent is to allow the regression to remove the effect of outlier scans
  by keeping some non-zero parameter from ``X_n``.
* We need to review these parameters to see if we really are detecting outliers.

****************
Where we are now
****************

See the `report_1` notebook in this repository.

************
Installation
************

To work on this repo::

    virtualenv consden
    source consden/bin/activate
    pip install numpy
    pip install -r requirements.txt

The commands here may well need to be edited to work correctly; I haven't
tested them recently, and I have done some refactoring since I last ran them.

You'll need the OpenFMRI ds114 dataset unpacked in ``./data/ds114`` - or
change the path value in ``ds114/__init__.py``, or link the location of the
data directory to ``./data`` at the same level as this README.

Preprocess the data with nipype, by::

    # Edit nipype_settings.py to your system
    python nipype_ds114_t2.py

Then run the constrained model with::

    python consden_ds114.py

Finally, you may be able to run some trivial validation metrics with::

    python consden_single_vo
    python intra_subject_corr.py
