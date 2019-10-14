# -*- coding: utf-8 -*-

"""Package where the GausianProcessMetaModel class is defined."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from btb.tuning.metamodels.base import BaseMetaModel


class GaussianProcessMetaModel(BaseMetaModel):
    """GaussianProcessMetaModel class.

    This class is responsible to create a ``GaussianProcessRegressor`` from the
    ``sklearn.gaussian_process`` package.

    Attributes:
        _MODEL_KWARGS (dict):
            Dictionary with the default ``kwargs`` for the ``GaussianProcessRegressor``
            instantiation.
        _MODEL_CLASS (type):
            Class to be instantiated and used for the ``self._model`` instantiation. In
            this case ``sklearn.gaussian_process.GaussainProcessRegressor``
    """
    _MODEL_CLASS = GaussianProcessRegressor

    _MODEL_KWARGS = {
        'normalize_y': True
    }

    def _predict(self, candidates):
        predictions = self._model.predict(candidates, return_std=True)
        return np.column_stack(predictions)
