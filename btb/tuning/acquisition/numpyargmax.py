# -*- coding: utf-8 -*-

"""Package where the NumpyArgMaxFunction class is defined."""
import numpy as np

from btb.tuning.acquisition.base import BaseAcquisitionFunction


class NumpyArgMaxFunction(BaseAcquisitionFunction):

    def _acquire(self, candidates, num_candidates=1):
        return list(reversed(np.argsort(candidates)))[:num_candidates]
