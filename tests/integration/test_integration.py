# -*- coding: utf-8 -*-

import random

from sklearn.datasets import load_boston as load_dataset
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from btb.benchmark import benchmark
from btb.benchmark.challenges import Rosenbrock
from btb.benchmark.tuners.btb import make_tuning_function
from btb.session import BTBSession
from btb.tuning import GPEiTuner, GPTuner, Tunable
from btb.tuning.hyperparams import (
    BooleanHyperParam, CategoricalHyperParam, FloatHyperParam, IntHyperParam)


def test_benchmark_rosenbrock():
    candidate = make_tuning_function(GPTuner)
    benchmark(candidate, challenges=Rosenbrock(), iterations=1)


def test_tunable_tuner():

    hyperparams = {
        'bhp': BooleanHyperParam(default=False),
        'chp': CategoricalHyperParam(choices=['a', 'b', None], default=None),
        'fhp': FloatHyperParam(min=0.1, max=1.0, default=0.5),
        'ihp': IntHyperParam(min=-1, max=1)
    }

    tunable = Tunable(hyperparams)

    tuner = GPEiTuner(tunable)

    for _ in range(10):
        proposed = tuner.propose(1)
        tuner.record(proposed, random.random())


def test_session():

    def build_model(name, hyperparameters):
        model_class = models[name]
        return model_class(random_state=0, **hyperparameters)

    def score_model(name, hyperparameters):
        model = build_model(name, hyperparameters)
        r2_scorer = make_scorer(r2_score)
        scores = cross_val_score(model, X_train, y_train, scoring=r2_scorer)
        return scores.mean()

    dataset = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.3, random_state=0)

    tunables = {
        'random_forest': {
            'n_estimators': {
                'type': 'int',
                'default': 2,
                'range': [1, 1000]
            },
            'max_features': {
                'type': 'str',
                'default': 'log2',
                'range': [None, 'auto', 'log2', 'sqrt']
            },
            'min_samples_split': {
                'type': 'int',
                'default': 2,
                'range': [2, 20]
            },
            'min_samples_leaf': {
                'type': 'int',
                'default': 2,
                'range': [1, 20]
            },
        },
        'extra_trees': {
            'n_estimators': {
                'type': 'int',
                'default': 2,
                'range': [1, 1000]
            },
            'max_features': {
                'type': 'str',
                'default': 'log2',
                'range': [None, 'auto', 'log2', 'sqrt']
            },
            'min_samples_split': {
                'type': 'int',
                'default': 2,
                'range': [2, 20]
            },
            'min_samples_leaf': {
                'type': 'int',
                'default': 2,
                'range': [1, 20]
            },
        }
    }

    models = {
        'random_forest': RandomForestRegressor,
        'extra_trees': ExtraTreesRegressor,
    }

    session = BTBSession(tunables, score_model, verbose=True)
    session.run(2)
