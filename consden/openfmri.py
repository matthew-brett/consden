""" Processing OpenFMRI datasets
"""
import os
from os.path import join as pjoin, abspath
import re

RE_SUB_NO = re.compile("sub(\d+)$")
RE_TASK_RUN_NO = re.compile("task(\d+)_run(\d+)$")

def path_to_subj(path):
    match = RE_SUB_NO.search(path)
    if match is None:
        return None
    return int(match.groups()[0])


def path_to_task_run(path):
    match = RE_TASK_RUN_NO.search(path)
    if match is None:
        return None, None
    return tuple(int(n) for n in match.groups())


def get_subjects(root_path):
    subjects = []
    for path in os.listdir(root_path):
        subj_no = path_to_subj(path)
        if subj_no is not None:
            subjects.append(Subject(root_path, subj_no))
    return subjects


def get_tasks_runs(task_run_path):
    tasks_runs = []
    for path in os.listdir(task_run_path):
        task_no, run_no = path_to_task_run(path)
        if task_no is not None:
            tasks_runs.append((task_no, run_no))
    return tasks_runs


class Subject(object):

    def __init__(self, root_path, subj_no):
        self.root_path = abspath(root_path)
        self.subj_no = subj_no
        self.path = pjoin(self.root_path, 'sub{0:03d}'.format(subj_no))
        self.bold_path = pjoin(self.path, 'BOLD')

    @property
    def tasks(self):
        return []

"""
A Subject has

* None or more Runs

A Run has a:

* task number
* run number
* None or more models

A Model has a:

* model number
* task number
* run number
* None or more onsets
"""

class Run(object):
    # task
    # models
    pass


class Task(object):
    # runs
    # models
    pass
