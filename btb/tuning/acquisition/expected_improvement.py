# -*- coding: utf-8 -*-

"""Package where the ExpectedImprovementAcquisition class is defined."""

import numpy as np
from scipy.stats import norm

from btb.tuning.acquisition.base import BaseAcquisition


class ExpectedImprovementAcquisition(BaseAcquisition):

    def _acquire(self, candidates, num_candidates=1):
        Phi = norm.cdf
        N = norm.pdf

        mu, sigma = candidates.T
        y_best = np.max(self.scores)

        z = (mu - y_best) / sigma

        ei = sigma * (z * Phi(z) + N(z))

        return self._get_max_candidates(ei, num_candidates)
