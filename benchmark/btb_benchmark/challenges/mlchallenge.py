# -*- coding: utf-8 -*-

"""Package where the MLChallenge class is defined."""

import logging
import os
from copy import deepcopy
from urllib.parse import urljoin

import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder

from btb_benchmark.challenges.challenge import Challenge
from btb_benchmark.challenges.datasets import get_dataset_names

BASE_DATASET_URL = 'https://atm-data.s3.amazonaws.com/'
BUCKET_NAME = 'atm-data'
LOGGER = logging.getLogger(__name__)


class MLChallenge(Challenge):
    """Machine Learning Challenge class.

    The MLChallenge class rerpresents a single ``machine learning challenge`` that can be used for
    benchmark.

    Args:
        model (class):
            Class of a machine learning estimator.
        dataset (str):
            Name or path to a dataset. If it's a name it will try to read it from
            https://atm-data.s3.amazonaws.com/
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
        max_rows (int):
            Maximum number of rows to use from each dataset. If ``None``, or if the
            given number is higher than the number of rows in the dataset, the entire
            dataset is used. Defaults to ``None``.
    """

    _data = None

    @classmethod
    def get_dataset_url(cls, name):
        if not name.endswith('.csv'):
            name = name + '.csv'

        return urljoin(BASE_DATASET_URL, name)

    @classmethod
    def get_all_challenges(cls, challenges='all'):
        """Return a list containing the instance of the datasets available.

        If ``challenges`` is provided, filter by those. Challenges can be a string
        representing the collection desired, all or openml100.

        Args:
            challenges (str or list):
                If string, a collection of datasets will be loaded from the module
                ``btb_benchmark.challenges.datasets``. If it's a list those will be
                the challenges attempted to be loaded.

        Returns:
            loaded_challenges (list):
                A list of ``btb_benchmark.challenges.mlchallenge.MLChallenge`` instances
                with the specified ``challenges``.
        """
        if isinstance(challenges, str):
            datasets = get_dataset_names(challenges)
        else:
            datasets = challenges

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
        if os.path.isdir(self.dataset):
            X = pd.read_csv(self.dataset)

        else:
            url = self.get_dataset_url(self.dataset)
            X = pd.read_csv(url)

        if self.max_rows:
            X = X.sample(min(len(X), self.max_rows))

        y = X.pop(self.target_column)

        if self.make_binary:
            y = y.iloc[0] == y

        if self.encode:
            ohe = OneHotEncoder(categories='auto')
            X = ohe.fit_transform(X)

        return X, y

    @property
    def data(self):
        if self._data is None:
            self._data = self.load_data()

        return self._data

    def __init__(self, dataset, model=None, target_column=None, encode=None,
                 tunable_hyperparameters=None, metric=None, model_defaults=None,
                 make_binary=None, stratified=None, cv_splits=5, cv_random_state=42,
                 cv_shuffle=True, metric_args={}, max_rows=None):

        self.model = model or self.MODEL
        self.dataset = dataset or self.DATASET
        self.target_column = target_column or self.TARGET_COLUMN
        self.model_defaults = model_defaults or self.MODEL_DEFAULTS
        self.make_binary = make_binary or self.MAKE_BINARY
        self.tunable_hyperparameters = tunable_hyperparameters or self.TUNABLE_HYPERPARAMETERS
        self.max_rows = max_rows

        if metric:
            self.metric = metric
            self.metric_args = metric_args

        else:
            # Allow to either write a metric method or assign a METRIC function
            self.metric = getattr(self, 'metric', self.__class__.METRIC)
            self.metric_args = getattr(self, 'metric_args', self.__class__.METRIC_ARGS)

        self.stratified = self.STRATIFIED if stratified is None else stratified
        # self.X, self.y = self.load_data()

        self.encode = self.ENCODE if encode is None else encode
        self.scorer = make_scorer(self.metric, **self.metric_args)

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
        X, y = self.data

        return cross_val_score(model, X, y, cv=self.cv, scoring=self.scorer).mean()

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.dataset)
