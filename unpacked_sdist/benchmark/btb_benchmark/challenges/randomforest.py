import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from btb_benchmark.challenges.mlchallenge import MLChallenge

LOGGER = logging.getLogger(__name__)


class RandomForestChallenge(MLChallenge):

    # TARGET
    TARGET_COLUMN = 'class'
    MAKE_BINARY = False

    # CROSS VALIDATE / SCORER
    METRIC = f1_score
    METRIC_ARGS = {'average': 'macro'}
    ENCODE = True
    STRATIFIED = True

    # MODEL
    MODEL = RandomForestClassifier
    MODEL_DEFAULTS = {
        'random_state': 0,
        'n_estimators': 100,
    }
    TUNABLE_HYPERPARAMETERS = {
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
