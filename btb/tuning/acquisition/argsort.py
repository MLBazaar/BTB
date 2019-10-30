# -*- coding: utf-8 -*-

"""Package where the ArgSortAcquisition class is defined."""
import numpy as np

from btb.tuning.acquisition.base import BaseAcquisition


class ArgSortAcquisition(BaseAcquisition):
    """ArgSortAcquisition class."""

    def _acquire(self, candidates, num_candidates=1):
        scores = candidates if len(candidates.shape) == 1 else candidates[:, 0]
        sorted_scores = list(reversed(np.argsort(scores)))
        return sorted_scores[:num_candidates]
