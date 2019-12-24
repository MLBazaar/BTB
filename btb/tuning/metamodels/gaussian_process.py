# -*- coding: utf-8 -*-

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from btb.tuning.metamodels.base import BaseMetaModel


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

    _MODEL_KWARGS_DEFAULT = {
        'normalize_y': True
    }

    def __init_metamodel__(self, length_scale=1):
        if self._model_kwargs is None:
            self._model_kwargs = {}

        self._model_kwargs['kernel'] = RBF(length_scale=length_scale)

    def _predict(self, candidates):
        predictions = self._model_instance.predict(candidates, return_std=True)
        return np.column_stack(predictions)
