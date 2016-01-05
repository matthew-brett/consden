""" Testing openfmri module
"""

from os.path import dirname, join as pjoin, isdir
from itertools import product

import numpy as np

from ..openfmri import (path_to_subj, path_to_task_run, get_subjects,
                        get_tasks_runs)

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
