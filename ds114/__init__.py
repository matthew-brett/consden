from __future__ import print_function

from os.path import (expanduser, split as psplit, join as pjoin, exists)

from consden.openfmri import get_subjects

# Init for ds114 package
TR = 2.5
# Where to find files on the filesystem
ROOT_PATH = expanduser('~/data/ds114')


SUBJECTS = get_subjects(ROOT_PATH)

def gen_models(subjects, task_nos=None, model_no=0):
    for subject in subjects:
        model = subject.models[model_no]
        for run_model in model.run_models:
            if task_nos and run_model.task_no not in task_nos:
                continue
            yield run_model
