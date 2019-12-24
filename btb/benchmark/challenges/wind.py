# -*- coding: utf-8 -*-

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from btb.benchmark.challenges.challenge import MLChallenge


class WindRFC(MLChallenge):
    # DATASET
    DATASET = 'wind.csv'
    TARGET_COLUMN = 'class'

    # CROSS VALIDATE / SCORER
    METRIC = f1_score
    ENCODE = True
    MAKE_BINARY = True
    STRATIFIED = True

    # MODEL
    MODEL = RandomForestClassifier
    MODEL_DEFAULTS = {'random_state': 0}
    TUNABLE_HYPERPARAMETERS = {
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


class WindSGDC(WindRFC):
    # MODEL
    MODEL = SGDClassifier
    TUNABLE_HYPERPARAMETERS = {
        "loss": {
            "type": "str",
            "default": "hinge",
            "range": [
                "log",
                "hinge",
                "modified_huber",
                "squared_hinge",
                "perceptron",
                "squared_loss",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive"
            ]
        },
        "penalty": {
            "type": "str",
            "default": None,
            "values": [
                None,
                "l2",
                "l1",
                "elasticnet"
            ]
        },
        "alpha": {
            "type": "float",
            "default": 0.0001,
            "values": [
                0.0001,
                1
            ]
        },
        "max_iter": {
            "type": "int",
            "default": 1000,
            "values": [
                1,
                5000
            ]
        },
        "tol": {
            "type": "float",
            "default": 1e-3,
            "values": [
                1e-3,
                1
            ]
        },
        "shuffle": {
            "type": "bool",
            "default": True,
        },
    }


class WindABC(WindRFC):
    # MODEL
    MODEL = AdaBoostClassifier
    TUNABLE_HYPERPARAMETERS = {
        "n_estimators": {
            "type": "int",
            "default": 50,
            "range": [
                1,
                500
            ]
        },
        "learning_rate": {
            "type": "float",
            "default": 1.0,
            "range": [
                1.0,
                10.0
            ]
        },
        "algorithm": {
            "type": "str",
            "default": "SAMME.R",
            "values": [
                "SAMME",
                "SAMME.R"
            ]
        }
    }
