# -*- coding: utf-8 -*-

"""Package where the BaseMetaModel class is defined."""

import numpy as np

from abc import ABCMeta, abstractmethod


class BaseMetaModel(metaclass=ABCMeta):

    _MODEL_CLASS = None
    _MODEL_KWARGS = None
    _model_kwargs = None
    _model = None

    def _init_model(self):
        """Create an instance of a self._MODEL_CLASS."""
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else dict()
        if self._model_kwargs:
            model_kwargs.update(self._model_kwargs)

        self._model = self._MODEL_CLASS(**model_kwargs)

    # def _init_meta_model(self):
    #     if self._MODEL_KWARGS:
    #         self._model_kwargs = self._MODEL_KWARGS.copy()
    #     else:
    #         self._model_kwargs = dict()

    def _fit(self, trials, scores):
        """Process params and scores and fit internal meta-model.

        Args:
            params (array-like):
                2D array-like with shape ``(n_trials, n_params)``.
            scores (array-like):
                2D array-like with shape ``(n_trials, 1)``.
        """
        self._init_model()
        self._model.fit(trials, scores)

    def _predict(self, candidates):
        """Predict performance for candidate params under this meta-model.

        Depending on the meta-model, the predictions could be point predictions or could also
        include a standard deviation at that point (like with a Gaussian Process meta-model).

        Args:
            candidates (array-like):
                2D array-like with shape ``(num_cadidates, num_params)``.
        Returns:
            predictions (array-like):
                2D array-like with shape ``(num_candidates, num_outputs)``.
        """
        return np.array(self._model.predict(candidates))
