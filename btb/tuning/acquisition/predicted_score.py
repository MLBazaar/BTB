# -*- coding: utf-8 -*-

import numpy as np

from btb.tuning.acquisition.base import BaseAcquisition


class PredictedScoreAcquisition(BaseAcquisition):

    def _acquire(self, candidates, num_candidates=1):
        candidates = candidates if len(candidates.shape) == 1 else candidates[:, 0]
        return self._get_max_candidates(candidates, num_candidates)
