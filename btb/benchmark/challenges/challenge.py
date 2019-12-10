# -*- coding: utf-8 -*-

"""Package where the Challenge class is defined."""

import os
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from urllib.parse import urljoin

import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
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
        """Return a dictionary with hyperparameters to be tuned."""
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

    Args:
        model (class):
            Class of a machine learning estimator.

        dataset (str):
            Name or path to a dataset. If it's a name it will try to read it from
            https://btb-data.s3.amazonaws.com/

        target_column (str):
            Name of the target column.

        encode (bool):
            Weither or not to encode ``X``.

        model_defaults (dict):
            Dictionary with default values for the model instantiation.

        make_binary (bool):
            Weither or not to make ``y`` binary.

        tunable_hyperparameters (dict):
            Dictionary representing the tunable hyperparameters for the challenge.

        scorer (callable):
            Scoring function. If ``None``, then the estimator's scoring function
            will be used in case there is otherwise the default ``cross_val_score`` function.
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

    def __init__(self, model=None, dataset=None, target_column=None,
                 encode=False, tunable_hyperparameters=None, scorer=None,
                 model_defaults=None, make_binary=None, cv_shuffle=True,
                 cv_splits=5, cv_random_state=42, stratified=True):

        self.model = model or self.MODEL
        self.dataset = dataset or self.DATASET
        self.target_column = target_column or self.TARGET_COLUMN
        self.model_defaults = model_defaults or self.MODEL_DEFAULTS
        self.make_binary = make_binary or self.MAKE_BINARY
        self.tunable_hyperparameters = tunable_hyperparameters or self.TUNABLE_HYPERPARAMETERS
        self.scorer = scorer or self.SCORER
        self.stratified = stratified
        self.X, self.y = self.load_data()

        self.encode = self.ENCODE if encode is None else encode

        if self.encode:
            ohe = OneHotEncoder()
            self.X = ohe.fit_transform(self.X)

    def get_tunable_hyperparameters(self):
        return deepcopy(self.tunable_hyperparameters)

    def evaluate(self, **hyperparams):
        """Apply cross validation to hyperparameter combination.

        Args:
            hyperparams (dict):
                A combination of ``self.tunable_hyperparams``.

        Returns:
            score (float):
                Returns the ``mean`` cross validated score.
        """
        if self.stratified and self.cv is None:
            self.cv = StratifiedKFold(
                shuffle=self.cv_shuffle,
                n_splits=self.cv_splits,
                random_state=self.cv_random_state
            )

        elif self.cv is None:
            self.cv = KFold(
                shuffle=self.cv_shuffle,
                n_splits=self.cv_splits,
                random_state=self.cv_random_state
            )

        if self._scorer is None:
            self._scorer = make_scorer(self.scorer)

        hyperparams.update((self.model_defaults or {}))
        model = self.model(**hyperparams)
        return cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self._scorer).mean()
