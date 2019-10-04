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
        self.trials = np.empty((0, self.tunable.K), dtype=np.float)
        self.scores = np.empty((0, 1), dtype=np.float)

    def _sample(self, num_proposals, allow_duplicates):
        """Generate a ``numpy.ndarray`` of valid proposals.

        Generates ``num_proposals`` of valid proposals by generating ``proposals`` until
        ``len(valid_proposals) == num_proposals`` or ``len(valid_proposals) == self.tunable.SC``
        being those ``proposals`` different from the ones that have been recorded.

        Args:
            num_proposals (int):
                Amount of proposals to generate.
            allow_duplicates (bool):
                If it's ``False``, the tuner will propose trials that are not recorded. Otherwise
                will generate trials that can be repeated.

        Returns:
            numpy.ndarray:
                A ``numpy.ndarray`` with shape ``(num_proposals, self.tunable.K)``.

        Raises:
            ValueError:
                If the unique amount of recorded trials is the same as the amount of combinations
                available for ``self.tunable``.

            ValueError:
                If the unique amount of recorded trials is the same as the amount of combinations
                available for ``self.tunable``.

        """
        if allow_duplicates:
            return self.tunable.sample(num_proposals)
        else:
            valid_proposals = list()
            trials_set = set(list(map(tuple, self.trials)))

            if len(trials_set) == self.tunable.SC:
                raise ValueError(
                    'All of the possible trials where recorded. Use ``allow_duplicates=True``'
                    'to keep generating trials.'
                )

            elif len(trials_set) + num_proposals > self.tunable.SC:
                raise ValueError(
                    'The maximum amount of new proposed combinations will exceed the amount of'
                    'possible combinations, either use `num_proposals={}` or'
                    '`allow_duplicates=True`.'.format(self.tunable.SC - len(trials_set))
                )

            while len(valid_proposals) != num_proposals:
                proposed = self.tunable.sample(num_proposals)
                proposed = list(map(tuple, proposed))

                if len(valid_proposals) > 0:
                    proposed = set(proposed) - set(valid_proposals)
                    if len(proposed) + len(valid_proposals) > num_proposals:
                        proposed = list(proposed)[:num_proposals - len(valid_proposals)]

                valid_proposals.extend(list(set(proposed) - trials_set))

                if len(valid_proposals) == self.tunable.SC:
                    return np.asarray(valid_proposals)

            return np.asarray(valid_proposals)

    @abstractmethod
    def _propose(self, num_proposals, allow_duplicates):
        """Generate ``num_proposals`` number of candidates.

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
            allow_duplicates (bool):
                If it's False, the tuner will propose trials that are not recorded. Otherwise
                will generate trials that can be repeated.

        Returns:
            dict or list:
                If ``num_proposals`` is 1, a ``dict`` will be returned containing the
                hyperparameter names and values. Otherwise, if ``num_proposals`` is bigger than 1,
                a list of such dicts is returned.

        Raises:
            ValueError:
                A ``ValueError`` exception is being produced if the amount of requested proposals
                is bigger than the possible combinations and ``allow_duplicates`` is ``False``.
        """
        if num_proposals > self.tunable.SC and not allow_duplicates:
            raise ValueError(
                'The number of samples is bigger than the combinations of the `tunable`.'
                'Use `allow_duplicates=True`, to generate more combinations.'
            )

        proposed = self._propose(num_proposals, allow_duplicates)
        hyperparameters = self.tunable.inverse_transform(proposed)
        hyperparameters = hyperparameters.to_dict(orient='records')

        if num_proposals == 1:
            hyperparameters = hyperparameters[0]

        return hyperparameters

    def record(self, trials, scores):
        """Record one or more ``trials`` with the associated ``score``.

        Records one or more ``trials`` with the associated ``score`` to it. The amount of trials
        must be equal to the amount of scores recived (and vice versa).

        Raises:
            ValueError:
                A ``ValueError`` exception is being produced if ``len(trials)`` is not equal to
                ``len(scores)``.
        """

        trials = self.tunable.transform(trials)
        scores = scores if isinstance(scores, (list, np.ndarray)) else [scores]

        if len(trials) != len(scores):
            raise ValueError('The amount of trials must be equal to the amount of scores.')

        self.trials = np.append(self.trials, trials, axis=0)
        self.scores = np.append(self.scores, scores)
