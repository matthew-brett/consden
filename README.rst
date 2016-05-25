######################
Constrained regression
######################

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
