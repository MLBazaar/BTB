# -*- coding: utf-8 -*-

"""Package where the BaseMetaModel class is defined."""

from abc import ABCMeta, abstractmethod


class BaseMetaModel(metaclass=ABCMeta):

    @abstractmethod
    def _init_model(self):
        """Create an instance of a model."""
        pass

    def __init__(self, *args, **kwargs):
        """Enables cooperative multiple inheritance."""
        super().__init__(*args, **kwargs)
        self._model = self._init_model()

    @abstractmethod
    def _fit(self, params, scores):
        """Process params and scores and fit internal meta-model.

        Args:
            params (array-like):
                2D array-like with shape ``(n_trials, n_params)``.
            scores (array-like):
                2D array-like with shape ``(n_trials, 1)``.
        """
        pass

    @abstractmethod
    def _predict(self, candidates):
        """Predict performance for candidate params under this meta-model.

        Depending on the meta-model, the predictions could be point predictions or could also
        include a standard deviation at that point (like with a Gaussian Process meta-model).

        Args:
            candidates (array-like):
                2D array-like with shape ``(n_cadidates, n_params)``.
        Returns:
            predictions (array-like):
                2D array-like with shape ``(n_candidates, n_outputs)``.
        """
        pass
