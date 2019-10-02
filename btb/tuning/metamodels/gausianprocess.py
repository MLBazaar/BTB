# -*- coding: utf-8 -*-

"""Package where the GausianProcessMetaModel class is defined."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from btb.tuning.metamodels.base import BaseMetaModel


class GaussianProcessMetaModel(BaseMetaModel):
    """GaussianProcessMetaModel class.

    Create an instance of ``GaussianProcessRegressor`` from the ``sklearn.gaussian_process``
    package.
    """

    def _init_model(self):
        """Create an instance of a GaussianProcessRegressor from sklearn."""
        return GaussianProcessRegressor(normalize_y=True)

    def _fit(self, params, scores):
        self._model.fit(params, scores)

    def _predict(self, candidates):
        y, stdev = self._model.predict(candidates, return_std=True)
        return np.array(list(zip(y, stdev)))
