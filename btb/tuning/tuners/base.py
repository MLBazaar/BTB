# -*- coding: utf-8 -*-

"""Package where the BaseTuner class is defined."""

from abc import abstractmethod

import numpy as np


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
        self.trials = list()
        self.results = list()

    def _sample(self, num_proposals, allow_duplicates):
        if allow_duplicates:
            return self.tunable.sample(num_proposals)
        else:
            valid = list()
            trials_list = list()
            trials_list.extend(map(tuple, self.trials))

            if len(self.trials) == self.tunable.SC:
                raise ValueError(
                    'All of the possible trials where recorded. Use allow_duplicates=True to keep'
                    'generating trials.'
                )

            while len(valid) != num_proposals or len(valid) == self.tunable.SC:
                proposed = self.tunable.sample(num_proposals)
                proposed = list(map(tuple, proposed))
                valid.append(set(proposed) - set(trials_list))

            return np.asarray(valid)

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

    def propose(self, num_proposals=1, allow_duplicates=False):
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

    def record(self, trials, results):
        """
        """

        trials = self.tunable.transform(trials)
        results = results if isinstance(results, (list, np.ndarray)) else [results]

        if len(trials) != len(results):
            raise ValueError('The amount of trials must be equal to the amount of results.')

        self.results.extend(results)
        self.trials.extend(trials)
