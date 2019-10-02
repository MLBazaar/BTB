# -*- coding: utf-8 -*-

"""Package where the UniformTuner class is defined."""

from btb.tuning.tuners.base import BaseTuner


class UniformTuner(BaseTuner):

    def _propose(self, num_proposals):
        """Returns a ``num_proposals`` number of samples."""
        return self.tunable.sample(num_proposals)
