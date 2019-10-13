# -*- coding: utf-8 -*-

"""Package where the ExpectedImprovementFunction class is defined."""

import numpy as np
from scipy.stats import norm

from btb.tuning.acquisition.numpyargmax import NumpyArgMaxFunction


class ExpectedImprovementFunction(NumpyArgMaxFunction):

    def _acquire(self, candidates, num_candidates=1):
        Phi = norm.cdf
        N = norm.pdf

        mu, sigma = candidates.T
        y_best = np.max(self._scores) if self.maximize else np.min(self._scores)

        z = (mu - y_best) / sigma

        ei = sigma * (z * Phi(z) + N(z))

        ei = np.column_stack((ei, np.zeros(len(ei))))
        return super()._acquire(candidates, num_candidates)
