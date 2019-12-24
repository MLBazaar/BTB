# -*- coding: utf-8 -*-

"""Package where the Challenge class is defined."""

import inspect
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

    if not name.endswith('.gz'):
        name = name + '.gz'

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

    def __repr__(self):
        args = inspect.getargspec(self.__init__)
        keys = args.args[1:]
        defaults = dict(zip(keys, args.defaults))
        instanced = {key: getattr(self, key) for key in keys}

        if defaults == instanced:
            return '{}()'.format(self.__class__.__name__)

        else:
            args = ', '.join(
                '{}={}'.format(key, value)
                for key, value in instanced.items()
            )

            return '{}({})'.format(self.__class__.__name__, args)


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
            Name of the target column in the dataset.

        encode (bool):
            Either or not to encode the dataset using ``sklearn.preprocessing.OneHotEncoder``.

        model_defaults (dict):
            Dictionary with default keyword args for the model instantiation.

        make_binary (bool):
            Either or not to make the target column binary.

        tunable_hyperparameters (dict):
            Dictionary representing the tunable hyperparameters for the challenge.

        metric (callable):
            Metric function. If ``None``, then the estimator's metric function
            will be used in case there is otherwise the default that ``cross_val_score`` function
            offers will be used.
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
                 encode=None, tunable_hyperparameters=None, metric=None,
                 model_defaults=None, make_binary=None, stratified=None,
                 cv_splits=5, cv_random_state=42, cv_shuffle=True):

        self.model = model or self.MODEL
        self.dataset = dataset or self.DATASET
        self.target_column = target_column or self.TARGET_COLUMN
        self.model_defaults = model_defaults or self.MODEL_DEFAULTS
        self.make_binary = make_binary or self.MAKE_BINARY
        self.tunable_hyperparameters = tunable_hyperparameters or self.TUNABLE_HYPERPARAMETERS

        if metric:
            self.metric = metric
        else:
            # Allow to either write a metric method or assign a METRIC function
            self.metric = getattr(self, 'metric', self.__class__.METRIC)

        self.stratified = self.STRATIFIED if stratified is None else stratified
        self.X, self.y = self.load_data()

        self.encode = self.ENCODE if encode is None else encode
        self.scorer = make_scorer(self.metric)

        if self.stratified:
            self.cv = StratifiedKFold(
                shuffle=cv_shuffle,
                n_splits=cv_splits,
                random_state=cv_random_state
            )
        else:
            self.cv = KFold(
                shuffle=cv_shuffle,
                n_splits=cv_splits,
                random_state=cv_random_state
            )

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
        hyperparams.update((self.model_defaults or {}))
        model = self.model(**hyperparams)
        return cross_val_score(model, self.X, self.y, cv=self.cv, scoring=self.scorer).mean()

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
