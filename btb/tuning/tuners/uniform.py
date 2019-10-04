# -*- coding: utf-8 -*-

"""Package where the UniformTuner class is defined."""

from btb.tuning.tuners.base import BaseTuner


class UniformTuner(BaseTuner):

    def _propose(self, num_proposals, allow_duplicates):
        """Returns a ``num_proposals`` number of samples."""
        return self._sample(num_proposals, allow_duplicates)
