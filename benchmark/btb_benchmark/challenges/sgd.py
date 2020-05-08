import logging

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from btb_benchmark.challenges.mlchallenge import MLChallenge

LOGGER = logging.getLogger(__name__)


class SGDChallenge(MLChallenge):

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
