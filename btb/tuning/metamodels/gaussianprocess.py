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

    def _fit(self, trials, scores):

        if not self.maximize:
            scores = [-score for score in scores]

        self._model = self._init_model()
        self._model.fit(trials, scores)

    def _predict(self, candidates):
        return np.array(self._model.predict(candidates))
