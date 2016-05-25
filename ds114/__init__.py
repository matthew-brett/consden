from __future__ import print_function

from os import environ
from os.path import (dirname, abspath, join as pjoin, isdir)

from consden.openfmri import get_subjects

# Init for ds114 package
TR = 2.5

# Where to find files on the filesystem
MY_PATH = dirname(__file__)
ROOT_PATH_REAL = abspath(pjoin(MY_PATH, '..', 'data', 'ds114'))
ROOT_PATH_TEST = abspath(pjoin(MY_PATH, '..', 'test-data', 'ds114'))
_testing = environ.get('CONSDEN_TEST') is not None
if isdir(ROOT_PATH_REAL) and not _testing:
    ROOT_PATH = ROOT_PATH_REAL
elif isdir(ROOT_PATH_TEST):
    ROOT_PATH = ROOT_PATH_TEST
else:
    msg = (ROOT_PATH_TEST + ' is not a directory' if _testing else
           "Neither real path {} nor test path {} is a directory"
           .format(ROOT_PATH_REAL, ROOT_PATH_TEST))
    raise RuntimeError(msg)

SUBJECTS = get_subjects(ROOT_PATH)

def gen_models(subjects, task_nos=None, model_no=0):
    for subject in subjects:
        model = subject.models[model_no]
        for run_model in model.run_models:
            if task_nos and run_model.task_no not in task_nos:
                continue
            yield run_model

from .ccana import get_sub_models, models_to_blocks, img_cc
