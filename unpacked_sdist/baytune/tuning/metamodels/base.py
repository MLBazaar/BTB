# -*- coding: utf-8 -*-

from abc import ABCMeta
from copy import deepcopy

import numpy as np


class BaseMetaModel(metaclass=ABCMeta):
    """BaseMetaModel class.

    BaseMetaModel class is an abstract representation of a ``MetaModel`` that after being
    fitted can predict the score that is being expected to be obtained with the hyperparameter
    configuration proposed.

    Attributes:
        _MODEL_CLASS (type):
            Class to be instantiated and to be fited and used to generate predictions. This
            attribute must be overriden by the child class in order to create an actual model.
        _MODEL_KWARGS (dict):
            Dictionary with the default ``kwargs`` to be given for the ``self._MODEL_CLASS``.
        _model_kwargs (dict):
            A dictionary that is used during instantiation of the ``BaseMetaModelTuner`` in
            order to be able to give or set arguments for the model.
        _model (object):
            Instance of ``self._MODEL_CLASS``, defaults to ``None``.
    """

    _MODEL_CLASS = None
    _MODEL_KWARGS_DEFAULT = None
    _model_kwargs = None
    _model_instance = None

    def __init_metamodel__(self, **kwargs):
        pass

    def _init_model(self):
        """Create an instance of a ``self._MODEL_CLASS``.

        Generate ``self._model_instance`` by using the corresponding ``kwargs`` for
        ``self._MODEL_CLASS`` provided by the user and ``self._MODEL_KWARGS_DEFAULT``.
        """
        if self._MODEL_KWARGS_DEFAULT is not None:
            model_kwargs = deepcopy(self._MODEL_KWARGS_DEFAULT)

        else:
            model_kwargs = {}

        if self._model_kwargs:
            model_kwargs.update(self._model_kwargs)

        self._model_instance = self._MODEL_CLASS(**model_kwargs)

    def _fit(self, trials, scores):
        """Fit the internal meta-model.

        Create a new instance of ``self._META_CLASS`` and fit it over ``trials`` and ``scores``.

        Args:
            params (array-like):
                2D array-like with shape ``(len(trials), n_params)``.
            scores (array-like):
                Array-like with shape ``(len(trials), 1)``.
        """
        self._init_model()
        self._model_instance.fit(trials, scores)

    def _predict(self, candidates):
        """Predict performance for given candidates.

        Predict the performance for the given candidates using the ``self._META_CLASS`` instance
        and it's method ``predict``. Depending on the meta-model, the predictions could be point
        predictions or could also include a standard deviation at that point (like with a
        Gaussian Process meta-model).

        Args:
            candidates (array-like):
                2D array-like with shape ``(num_cadidates, num_params)``.
        Returns:
            predictions (array-like):
                2D array-like with shape ``(num_candidates, num_outputs)``.
        """
        return np.array(self._model_instance.predict(candidates))
