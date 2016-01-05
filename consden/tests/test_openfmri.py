""" Testing openfmri module
"""

from os.path import dirname, join as pjoin, isdir, isfile
from itertools import product

import numpy as np

from ..openfmri import (path_to_subj, path_to_task_run, get_subjects,
                        get_tasks_runs, get_model_nos, get_conditions)

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


DS114 = pjoin(dirname(__file__), 'ds114')
SUB1 = pjoin(DS114, 'sub001')


def test_path_to_subj():
    assert_equal(path_to_subj(pjoin('data', 'ds114', 'sub009')), 9)
    assert_equal(path_to_subj(pjoin('data', 'ds114', 'sub_009')), None)


def test_path_to_task_run():
    assert_equal(path_to_task_run(pjoin('BOLD', 'task001_run001')), (1, 1))
    assert_equal(path_to_task_run(pjoin('BOLD', 'task001run001')),
                 (None, None))


def test_get_subjects():
    subjects = get_subjects(DS114)
    assert_equal(len(subjects), 2)
    for subj_no, subject in zip(range(1, 3), subjects):
        assert_true(isdir(subject.path))
        assert_true(subject.path.endswith('sub00{0}'.format(subj_no)))
        assert_equal(subject.subj_no, subj_no)


def test_get_tasks_runs():
    for path in (pjoin(SUB1, 'BOLD'),
                 pjoin(SUB1, 'model', 'model001', 'onsets'),
                 pjoin(SUB1, 'model', 'model002', 'onsets')):
        tasks_runs = get_tasks_runs(path)
        assert_equal(tasks_runs, list(product(range(1, 6), range(1, 3))))


def test_get_model_nos():
    assert_equal(get_model_nos(pjoin(SUB1, 'model')), (1, 2))


def test_runs():
    subject = get_subjects(DS114)[0]
    for run in subject.runs:
        assert_true(isfile(run.get_bold_fname()))


def test_models():
    subject = get_subjects(DS114)[0]
    models = subject.models
    assert_equal(len(models), 2)
    # Number of conditions per task, for both models
    exp_lens = {1: 1, 2: 1, 3: 3, 4:1, 5:5}
    for model in models:
        for model_run in model.run_models:
            assert_true(isdir(model_run.path))
            conds = model_run.conditions
            assert_equal(len(conds), exp_lens[model_run.task_no])
            assert_true(isfile(model_run.run.get_bold_fname()))
