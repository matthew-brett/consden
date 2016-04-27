""" Processing OpenFMRI datasets
"""
import os
from os.path import join as pjoin, abspath
import re

import numpy as np

RE_SUB_NO = re.compile(r"sub(\d+)$")
RE_TASK_RUN_NO = re.compile(r"task(\d+)_run(\d+)$")
RE_MODEL_NO = re.compile(r"model(\d+)$")
RE_COND_NO = re.compile(r"cond(\d+).txt$")


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


def get_model_nos(models_path):
    models = []
    for fname in os.listdir(models_path):
        model_match = RE_MODEL_NO.match(fname)
        if model_match is None:
            continue
        models.append(int(model_match.groups()[0]))
    return tuple(models)


def get_conditions(model_run_path):
    conditions = []
    for fname in os.listdir(model_run_path):
        cond_match = RE_COND_NO.match(fname)
        if cond_match is None:
            continue
        ons_dur_amp = np.loadtxt(pjoin(model_run_path, fname))
        conditions.append((int(cond_match.groups()[0]), ons_dur_amp))
    return conditions


class Run(object):

    def __init__(self, subject, task_no, run_no):
        self.subject = subject
        self.task_no = task_no
        self.run_no = run_no

    @property
    def path(self):
        return pjoin(self.subject.path, 'BOLD',
                     'task{0:03d}_run{1:03d}'.format(
                         self.task_no, self.run_no))

    def get_bold_fname(self, prefix=''):
        return pjoin(self.path, prefix + 'bold.nii.gz')


class RunModel(object):

    def __init__(self, run, models):
        self.run = run
        self.models = models

    @property
    def model_no(self):
        return self.models.model_no

    @property
    def task_no(self):
        return self.run.task_no

    @property
    def run_no(self):
        return self.run.run_no

    @property
    def path(self):
        return pjoin(self.models.path,
                     'onsets',
                     'task{0:03d}_run{1:03d}'.format(
                         self.task_no, self.run_no))

    @property
    def conditions(self):
        return get_conditions(self.path)


class Models(object):
    run_model_maker = RunModel

    def __init__(self, subject, model_no, runs):
        self.subject = subject
        self.model_no = model_no
        self.runs = runs

    @property
    def path(self):
        return pjoin(self.subject.path,
                     'model',
                     'model{0:03d}'.format(self.model_no))

    @property
    def run_models(self):
        rms = []
        for run in self.runs:
            rms.append(self.run_model_maker(run, self))
        return rms


class Subject(object):
    run_maker = Run
    models_maker = Models

    def __init__(self, root_path, subj_no):
        self.root_path = abspath(root_path)
        self.subj_no = subj_no

    @property
    def path(self):
        return pjoin(self.root_path, 'sub{0:03d}'.format(self.subj_no))

    @property
    def runs(self):
        runs = []
        for task_no, run_no in get_tasks_runs(pjoin(self.path, 'BOLD')):
            runs.append(self.run_maker(self, task_no, run_no))
        return runs

    @property
    def models(self):
        models = []
        runs = self.runs
        for model_no in get_model_nos(pjoin(self.path, 'model')):
            models.append(self.models_maker(self, model_no, runs))
        return models
