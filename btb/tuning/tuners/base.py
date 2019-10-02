# -*- coding: utf-8 -*-

"""Package where the BaseTuner class is defined."""

from abc import abstractmethod


class BaseTuner:
    """BaseTuner class.

    BaseTuner class is the abstract representation of a tuner that is not based on a model.

    Attributes:
        tunable (btb.tuning.tunable.Tunable):
            Instance of a tunable class containing hyperparameters to be tuned.
    Args:
        tunable (btb.tuning.tunable.Tunable):
            Instance of a tunable class containing hyperparameters to be tuned.
    """

    def __init__(self, tunable):
        self.tunable = tunable

    @abstractmethod
    def _propose(self, num_proposals):
        """Return ``num_proposals`` number of candidates.

        Args:
            num_proposals (int):
                Number of candidates to create.

        Returns:
            numpy.ndarray:
                It returns ``numpy.ndarray`` with shape
                ``(num_proposals, len(tunable.hyperparameters)``.

        """
        pass

    def propose(self, num_proposals=1):
        """Propose (one or more) new hyperparameter configurations.

        Call the implemented ``_propose`` method and convert the returned data in to hyperparameter
        space.

        Args:
            num_proposals (int):
                Number of candidates to create.
        Returns:
            dict or list:
                If ``num_proposals`` is 1, a ``dict`` will be returned containing the
                hyperparameter names and values.
                Otherwise, if ``num_proposals`` is bigger than 1, a list of such dicts is returned.
                """
        proposed = self._propose(num_proposals)
        hyperparameters = self.tunable.inverse_transform(proposed)
        hyperparameters = hyperparameters.to_dict(orient='records')

        if num_proposals == 1:
            hyperparameters = hyperparameters[0]

        return hyperparameters
