# -*- coding: utf-8 -*-

"""Package where the UniformTuner class is defined."""

from btb.tuning.tuners.base import BaseTuner


class UniformTuner(BaseTuner):

    def _propose(self, num_proposals, allow_duplicates):
        """Generate ``num_proposals`` number of candidates.

        Args:
            num_proposals (int):
                Number of candidates to create.
            allow_duplicates (bool):
                If it's ``False``, the tuner will propose trials that are not recorded. Otherwise
                will generate trials that can be repeated.

        Returns:
            numpy.ndarray:
                It returns ``numpy.ndarray`` with shape
                ``(num_proposals, len(self.tunable.hyperparameters)``.
        """
        return self._sample(num_proposals, allow_duplicates)
