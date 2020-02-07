from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

from btb.benchmark.challenges.challenge import MLChallenge


class BostonRFR(MLChallenge):
    """
    This dataset contains information collected by the U.S Census Service
    concerning housing in the area of Boston. It was obtained from the
    StatLib archive http://lib.stat.cmu.edu/datasets/boston

    The prediction task is to determine the price of a house based on the
    location and other features that the dataset provides.
    """

    # DATSET
    DATASET = 'boston.csv'
    TARGET_COLUMN = 'medv'
    STRATIFIED = False

    # CROSS VALIDATE / SCORER
    METRIC = r2_score
    ENCODE = True
    MAKE_BINARY = False

    # MODEL
    MODEL = RandomForestRegressor
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


class BostonABR(BostonRFR):
    # MODEL
    MODEL = AdaBoostRegressor
    MODEL_DEFAULTS = {'random_state': 0}
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


class BostonBR(BostonRFR):
    # MODEL
    MODEL = BaggingRegressor
    MODEL_DEFAULTS = {'bootstrap': True, 'random_state': 0}
    TUNABLE_HYPERPARAMETERS = {
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
