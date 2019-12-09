# -*- coding: utf-8 -*-

"""Package where the Challenge class is defined."""

import os
from abc import ABCMeta, abstractmethod
from urllib.parse import urljoin

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

BTB_DATA_URL = 'https://btb-data.s3.amazonaws.com/'


def _get_dataset_url(name):
    if not name.endswith('.gzip'):
        name = name + '.gzip'

    return urljoin(BTB_DATA_URL, name)


class Challenge(metaclass=ABCMeta):
    """Challenge class.

    The Challenge class represents a single ``challenge`` that can be used for benchmark.
    """

    @abstractmethod
    def get_tunable_hyperparameters(self):
        """Create the ``hyperparameters`` and return the ``tunable`` created with them.

        Returns:
            ``btb.tuning.Tunable``:
                A ``Tunable`` instance to be used to tune the ``self.score`` method.
        """
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Perform evaluation for the given ``arguments``.

        This method will score a result with a given configuration, then return the score obtained
        for those ``arguments``.
        """
        pass


class MLChallenge(Challenge):
    """Machine Learning Challenge class.

    The MLChallenge class rerpresents a single ``machine learning challenge`` that can be used for
    benchmark.
    """

    def load_data(self):
        """Load ``X`` and ``y`` over which to perform fit and evaluate."""
        if os.path.isdir(self.dataset):
            X = pd.read_csv(self.dataset)

        else:
            url = _get_dataset_url(self.dataset)
            X = pd.read_csv(url, compression='gzip')

        y = X.pop(self.target_column)

        if self.make_binary:
            y = y.iloc[0] == y

        return X, y

    def __init__(
        self,
        model=None,
        dataset=None,
        target_column=None,
        encode=False,
        tunable_hyperparameters=None,
        cv=None,
        scorer=None,
        model_defaults=None,
        make_binary=None
    ):

        self.model = model or self.MODEL
        self.dataset = dataset or self.DATASET
        self.target_column = target_column or self.TARGET_COLUMN
        self.encode = encode or self.ENCODE
        self.model_defaults = model_defaults or self.MODEL_DEFAULTS
        self.make_binary = make_binary or self.MAKE_BINARY
        self.tunable_hyperparameters = tunable_hyperparameters or self.TUNABLE_HYPERPARAMETERS
        self.cv = cv or self.CV
        self.scorer = scorer or self.SCORER

        self.X, self.y = self.load_data()

        if self.encode:
            ohe = OneHotEncoder()
            self.X = ohe.fit_transform(self.X)

    def get_tunable_hyperparameters(self):
        return self.tunable_hyperparameters

    def evaluate(self, **hyperparams):
        """Apply cross validation to hyperparameter combination.

        Args:
            hyperparams (dict):
                A combination of ``self.tunable_hyperparams``.

        Returns:
            score (float):
                Returns the ``mean`` cross validated score.
        """
        hyperparams.update((self.model_defaults or {}))
        model = self.model(**hyperparams)
        return cross_val_score(model, self.X, self.y, cv=self.cv).mean()
