# -*- coding: utf-8 -*-

from unittest.mock import patch

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from btb.benchmark.challenges.census import CensusABC, CensusRFC, CensusSGDC


@patch('btb.benchmark.challenges.challenge.OneHotEncoder')
@patch('btb.benchmark.challenges.challenge.MLChallenge.load_data')
def test_get_tunable_hyperparameters(mock_load_data, mokc_ohe):

    # setup
    mock_load_data.return_value = ('X', 'y')

    censusabc = CensusABC()
    censusrfc = CensusRFC()
    censussgdc = CensusSGDC()

    # run
    abc_tunables = censusabc.get_tunable_hyperparameters()
    rfc_tunables = censusrfc.get_tunable_hyperparameters()
    sgdc_tunables = censussgdc.get_tunable_hyperparameters()

    # assert
    expected_abc_tunables = {
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

    expected_sgdc_tunables = {
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
        }
    }

    expected_rfc_tunables = {
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
        }
    }

    assert expected_abc_tunables == abc_tunables
    assert expected_sgdc_tunables == sgdc_tunables
    assert expected_rfc_tunables == rfc_tunables


@patch('btb.benchmark.challenges.challenge.make_scorer')
@patch('btb.benchmark.challenges.challenge.OneHotEncoder')
@patch('btb.benchmark.challenges.challenge.MLChallenge.load_data')
def test_class_attributes(mock_load_data, mokc_ohe, mock_make_scorer):

    # setup
    mock_load_data.return_value = ('X', 'y')

    # run
    censusabc = CensusABC()
    censusrfc = CensusRFC()
    censussgdc = CensusSGDC()

    # assert

    # ABC
    assert censusabc.model == AdaBoostClassifier
    assert censusabc.dataset == 'census.csv'
    assert censusabc.target_column == 'income'
    assert censusabc.model_defaults == {'random_state': 0}
    assert censusabc.make_binary
    assert censusabc.metric == f1_score
    assert censusabc.stratified

    # RFC
    assert censusrfc.model == RandomForestClassifier
    assert censusrfc.dataset == 'census.csv'
    assert censusrfc.target_column == 'income'
    assert censusrfc.model_defaults == {'random_state': 0}
    assert censusrfc.make_binary
    assert censusrfc.metric == f1_score
    assert censusrfc.stratified

    # SGDC
    assert censussgdc.model == SGDClassifier
    assert censussgdc.dataset == 'census.csv'
    assert censussgdc.target_column == 'income'
    assert censussgdc.model_defaults == {'random_state': 0}
    assert censussgdc.make_binary
    assert censussgdc.metric == f1_score
    assert censussgdc.stratified
