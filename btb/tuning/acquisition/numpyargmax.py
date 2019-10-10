# -*- coding: utf-8 -*-

"""Package where the NumpyArgMaxFunction class is defined."""
import numpy as np

from btb.tuning.acquisition.base import BaseAcquisitionFunction


class NumpyArgMaxFunction(BaseAcquisitionFunction):

    def _acquire(self, candidates, num_candidates=1):
        scores = candidates[:, 0]
        sorted_scores = list(reversed(np.argsort(scores)))
        return sorted_scores[:num_candidates]
