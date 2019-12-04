# -*- coding: utf-8 -*-

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder

from btb.benchmark.challenges.challenge import MLChallenge

DEFAULT_CV = StratifiedKFold(shuffle=True, n_splits=5)
DEFAULT_SCORER = make_scorer(f1_score)


class CensusRF(MLChallenge):

    MODEL_CLASS = RandomForestClassifier
    DEFAULT_HYPERPARAMS = {
        'random_state': 0,
    }

    @staticmethod
    def load_data():
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
        path = os.path.join(path, 'census.tar.gz')

        if not os.path.exists(path):
            raise ValueError('Census dataset has not been found on {}'.format(path))

        X = pd.read_csv(path, compression='gzip')
        y = X.pop('income') == ' >50K'
        return X, y

    def __init__(self, cv=DEFAULT_CV, scorer=DEFAULT_SCORER):
        self.X, self.y = self.load_data()

        ohe = OneHotEncoder()

        self.X = ohe.fit_transform(self.X)
        self.scorer = scorer
        self.cv = cv

    def get_tunable_hyperparameters(self):
        return {
            "n_estimators": {
                "type": "int",
                "default": 10,
                "range": [
                    1,
                    500
                ]
            },
            "criterion": {
                "type": "str",
                "default": "gini",
                "values": [
                    "entropy",
                    "gini"
                ]
            },
            "max_features": {
                "type": "str",
                "default": None,
                "values": [
                    None,
                    "auto",
                    "log2",
                    "sqrt"
                ]
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "range": [
                    2,
                    100
                ]
            },
            "min_samples_leaf": {
                "type": "int",
                "default": 1,
                "range": [
                    1,
                    100
                ]
            },
            "min_weight_fraction_leaf": {
                "type": "float",
                "default": 0.0,
                "range": [
                    0.0,
                    0.5
                ]
            },
            "min_impurity_decrease": {
                "type": "float",
                "default": 0.0,
                "range": [
                    0.0,
                    1000.0
                ]
            },
        }

    def evaluate(self, **hyperparams):
        hyperparams.update(self.DEFAULT_HYPERPARAMS)
        model = self.MODEL_CLASS(**hyperparams)
        return cross_val_score(model, self.X, self.y, cv=self.cv).mean()
