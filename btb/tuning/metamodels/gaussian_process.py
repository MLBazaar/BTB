# -*- coding: utf-8 -*-

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

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

    def _predict(self, candidates):
        predictions = self._model.predict(candidates, return_std=True)
        return np.column_stack(predictions)
