# -*- coding: utf-8 -*-

"""Package where the NumpyArgSortFunction class is defined."""
import numpy as np

from btb.tuning.acquisition.base import BaseAcquisitionFunction


class NumpyArgSortFunction(BaseAcquisitionFunction):
    """NumpyArgSortFunction class."""

    def _acquire(self, candidates, num_candidates=1):
        scores = candidates if len(candidates.shape) == 1 else candidates[:, 0]
        sorted_scores = np.argsort(scores)
        sorted_scores = list(reversed(sorted_scores)) if self.maximize else list(sorted_scores)
        return sorted_scores[:num_candidates]
