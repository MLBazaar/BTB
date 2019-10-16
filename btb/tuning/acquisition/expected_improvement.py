# -*- coding: utf-8 -*-

"""Package where the ExpectedImprovementAcquisition class is defined."""

import numpy as np
from scipy.stats import norm

from btb.tuning.acquisition.argsort import ArgSortAcquisition


class ExpectedImprovementAcquisition(ArgSortAcquisition):

    def _acquire(self, candidates, num_candidates=1):
        Phi = norm.cdf
        N = norm.pdf

        mu, sigma = candidates.T
        y_best = np.max(self._scores)

        z = (mu - y_best) / sigma

        ei = sigma * (z * Phi(z) + N(z))

        return super()._acquire(ei, num_candidates)
