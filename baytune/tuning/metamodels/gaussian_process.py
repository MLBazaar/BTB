# -*- coding: utf-8 -*-

import numpy as np
import scipy
from copulas import EPSILON
from copulas.univariate import Univariate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from baytune.tuning.metamodels.base import BaseMetaModel


class GaussianProcessMetaModel(BaseMetaModel):
    """GaussianProcessMetaModel class.

    This class represents a meta-model using an underlying ``GaussianProcessRegressor`` from
    ``sklearn.gaussian_process``.

    Attributes:
        _MODEL_KWARGS (dict):
            Dictionary with the default ``kwargs`` for the ``GaussianProcessRegressor``
            instantiation.
        _MODEL_CLASS (type):
            Class to be instantiated and used for the ``self._model`` instantiation. In
            this case ``sklearn.gaussian_process.GaussainProcessRegressor``
    """

    _MODEL_CLASS = GaussianProcessRegressor

    _MODEL_KWARGS_DEFAULT = {"normalize_y": True}

    def __init_metamodel__(self, length_scale=1):
        if self._model_kwargs is None:
            self._model_kwargs = {}

        self._model_kwargs["kernel"] = RBF(length_scale=length_scale)

    def _predict(self, candidates):
        predictions = self._model_instance.predict(candidates, return_std=True)
        return np.column_stack(predictions)


class GaussianCopulaProcessMetaModel(GaussianProcessMetaModel):
    """GaussianCopulaProcessMetaModel class.

    This class represents a meta-model using an underlying ``GaussianProcessRegressor`` from
    ``sklearn.gaussian_process`` applying ``copulas.univariate.Univariate`` transformations
    to the input data and afterwards reverts it for the predictions.

    During the ``fit`` process, this metamodel trains a univariate copula for each
    hyperparameter to then compute the cumulative distribution of these. Once the cumulative
    distribution has been calculated, we calculate the inverse of the normal cumulative
    distribution using ``scipy.stats.norm`` and use these transformations to train the
    ``GaussianProcessRegressor`` model.

    When predicting the output value, an inverse of the normal cumulative distribution is
    computed to the normal cumulative distribution, using the previously trained univariate
    copula with the input data of the score.

    Attributes:
        _MODEL_KWARGS (dict):
            Dictionary with the default ``kwargs`` for the ``GaussianProcessRegressor``
            instantiation.
        _MODEL_CLASS (type):
            Class to be instantiated and used for the ``self._model`` instantiation. In
            this case ``sklearn.gaussian_process.GaussainProcessRegressor``
    """

    def _transform(self, trials):
        transformed = []
        for column, distribution in zip(trials.T, self._distributions):
            transformed.append(
                scipy.stats.norm.ppf(
                    distribution.cdf(column).clip(0 + EPSILON, 1 - EPSILON)
                )
            )

        return np.column_stack(transformed)

    def _fit(self, trials, scores):
        self._distributions = []
        for column in trials.T:
            distribution = Univariate()
            distribution.fit(column)
            self._distributions.append(distribution)

        distribution = Univariate()
        distribution.fit(scores)
        self._score_distribution = distribution

        trans_trials = self._transform(trials)
        trans_scores = scipy.stats.norm.ppf(
            self._score_distribution.cdf(scores).clip(0 + EPSILON, 1 - EPSILON)
        )

        super()._fit(trans_trials, trans_scores)

    def _predict(self, candidates):
        trans_candidates = self._transform(candidates)
        predicted = super()._predict(trans_candidates)
        cdf = scipy.stats.norm.cdf(predicted)
        if len(cdf.shape) == 2:
            cdf = cdf[:, 0]
        return self._score_distribution.ppf(cdf)
