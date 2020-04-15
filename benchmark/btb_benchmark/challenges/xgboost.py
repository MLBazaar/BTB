import logging

from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from btb_benchmark.challenges.challenge import MLChallenge

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
    MODEL_DEFAULTS = {'random_state': 0}
    MODEL = XGBClassifier
    TUNABLE_HYPERPARAMETERS = {
        "n_estimators": {
            "type": "int",
            "default": 100,
            "range": [
                10,
                1000
            ]
        },
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
