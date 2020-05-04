import logging

from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from btb_benchmark.challenges.mlchallenge import MLChallenge

LOGGER = logging.getLogger(__name__)


class XGBoostChallenge(MLChallenge):

    # TARGET
    TARGET_COLUMN = 'class'
    MAKE_BINARY = False

    # CROSS VALIDATE / SCORER
    METRIC = f1_score
    METRIC_ARGS = {'average': 'macro'}
    ENCODE = True
    STRATIFIED = True

    # MODEL
    MODEL_DEFAULTS = {
        'random_state': 0,
        'n_estimators': 100
    }
    MODEL = XGBClassifier
    TUNABLE_HYPERPARAMETERS = {
        "max_depth": {
            "type": "int",
            "default": 3,
            "range": [
                3,
                10
            ]
        },
        "learning_rate": {
            "type": "float",
            "default": 0.1,
            "range": [
                0,
                1
            ]
        },
        "gamma": {
            "type": "float",
            "default": 0,
            "range": [
                0,
                1
            ]
        },
        "min_child_weight": {
            "type": "int",
            "default": 1,
            "range": [
                1,
                10
            ]
        }
    }
