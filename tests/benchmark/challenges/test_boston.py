# -*- coding: utf-8 -*-

from unittest.mock import patch

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

from btb.benchmark.challenges.boston import BostonABR, BostonBR, BostonRFR


@patch('btb.benchmark.challenges.challenge.OneHotEncoder')
@patch('btb.benchmark.challenges.challenge.MLChallenge.load_data')
def test_get_tunable_hyperparameters(mock_load_data, mock_ohe):

    # setup
    mock_load_data.return_value = ('X', 'y')

    bostonabr = BostonABR()
    bostonbr = BostonBR()
    bostonrfr = BostonRFR()

    # run
    abr_tunable_hyperparameters = bostonabr.get_tunable_hyperparameters()
    br_tunable_hyperparameters = bostonbr.get_tunable_hyperparameters()
    rfr_tunable_hyperparameters = bostonrfr.get_tunable_hyperparameters()

    # assert
    expected_abr_hyperparameters = {
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
        "loss": {
            "type": "str",
            "default": "linear",
            "values": [
                "linear",
                "square",
                "exponential"
            ]
        }
    }

    expected_br_hyperparameters = {
        "n_estimators": {
            "type": "int",
            "default": 10,
            "range": [
                1,
                500
            ]
        },
        "max_samples": {
            "type": "int",
            "default": 1,
            "range": [
                1,
                100
            ]
        },
    }

    expected_rfr_hyperparameters = {
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
            "default": "mse",
            "values": [
                "mse",
                "mae"
            ]
        },
        "max_features": {
            "type": "str",
            "default": "auto",
            "range": [
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
                1000
            ]
        },
        "min_samples_leaf": {
            "type": "int",
            "default": 1,
            "range": [
                1,
                1000
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
                10.0
            ]
        }
    }

    assert expected_abr_hyperparameters == abr_tunable_hyperparameters
    assert expected_br_hyperparameters == br_tunable_hyperparameters
    assert expected_rfr_hyperparameters == rfr_tunable_hyperparameters


@patch('btb.benchmark.challenges.challenge.make_scorer')
@patch('btb.benchmark.challenges.challenge.OneHotEncoder')
@patch('btb.benchmark.challenges.challenge.MLChallenge.load_data')
def test_class_attributes(mock_load_data, mokc_ohe, mock_make_scorer):

    # setup
    mock_load_data.return_value = ('X', 'y')

    # run
    bostonabr = BostonABR()
    bostonbr = BostonBR()
    bostonrfr = BostonRFR()

    # assert

    # ABR
    assert bostonabr.model == AdaBoostRegressor
    assert bostonabr.dataset == 'boston.csv'
    assert bostonabr.target_column == 'medv'
    assert bostonabr.model_defaults == {'random_state': 0}
    assert not bostonabr.make_binary
    assert bostonabr.metric == r2_score
    assert not bostonabr.stratified

    # BR
    assert bostonbr.model == BaggingRegressor
    assert bostonbr.dataset == 'boston.csv'
    assert bostonbr.target_column == 'medv'
    assert bostonbr.model_defaults == {'bootstrap': True, 'random_state': 0}
    assert not bostonbr.make_binary
    assert bostonbr.metric == r2_score
    assert not bostonbr.stratified

    # RFR
    assert bostonrfr.model == RandomForestRegressor
    assert bostonrfr.dataset == 'boston.csv'
    assert bostonrfr.target_column == 'medv'
    assert bostonrfr.model_defaults == {'random_state': 0}
    assert not bostonrfr.make_binary
    assert bostonrfr.metric == r2_score
    assert not bostonrfr.stratified
