# -*- coding: utf-8 -*-

"""Package where the GausianProcessMetaModel class is defined."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

from btb.tuning.metamodels.base import BaseMetaModel


class GaussianProcessMetaModel(BaseMetaModel):
    """GaussianProcessMetaModel class.

    Create an instance of ``GaussianProcessRegressor`` from the ``sklearn.gaussian_process``
    package.
    """
    _MODEL_KWARGS = {
        'normalize_y': True
    }
    _MODEL_CLASS = GaussianProcessRegressor

    def _predict(self, candidates):
        predictions = self._model.predict(candidates, return_std=True)
        return np.column_stack(predictions)


class RandomForestMetaModel(BaseMetaModel):
    _MODEL_CLASS = RandomForestRegressor
    _MODEL_KWARGS = {'n_estimators': 100}

    def _predict(self, candidates):
        predictions = self._model.predict(candidates)
        return np.column_stack([predictions, np.zeros(len(predictions))])
