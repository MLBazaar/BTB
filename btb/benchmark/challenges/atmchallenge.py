import logging
from urllib.parse import urljoin

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from btb.benchmark.challenges.challenge import MLChallenge

ATM_DATA_URL = 'http://atm-data.s3.amazonaws.com/'
LOGGER = logging.getLogger(__name__)


class ATMChallenge(MLChallenge):

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

    @classmethod
    def get_atm_dataset_url(cls, name):
        if not name.endswith('.csv'):
            name = name + '.csv'

        return urljoin(ATM_DATA_URL, name)

    @classmethod
    def get_available_datasets(cls):
        client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        available_datasets = [
            obj['Key']
            for obj in client.list_objects(Bucket='atm-data')['Contents']
            if obj['Key'] != 'index.html'
        ]

        return available_datasets

    @classmethod
    def get_all_challenges(cls, challenges=None):
        datasets = challenges or cls.get_available_datasets()
        loaded_challenges = []
        for dataset in datasets:
            try:
                loaded_challenges.append(cls(dataset))
                LOGGER.info('Dataset %s loaded', dataset)
            except Exception as ex:
                LOGGER.warn('Dataset: %s could not be loaded. Error: %s', dataset, ex)

        LOGGER.info('%s / %s datasets loaded.', len(loaded_challenges), len(datasets))

        return loaded_challenges

    def load_data(self):
        """Load ``X`` and ``y`` over which to perform fit and evaluate."""
        url = self.get_atm_dataset_url(self.dataset)
        X = pd.read_csv(url)

        y = X.pop(self.target_column)

        if self.make_binary:
            y = y.iloc[0] == y

        return X, y

    def __init__(self, dataset, **kwargs):
        kwargs['dataset'] = dataset
        super().__init__(**kwargs)

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.dataset)
