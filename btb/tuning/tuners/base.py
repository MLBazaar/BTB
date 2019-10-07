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
        trials (numpy.ndarray):
            A ``numpy.ndarray`` with shape ``(n, self.tunable.K)`` where ``n`` is the number of
            trials recorded.
        scores (numpy.ndarray):
            A ``numpy.ndarray`` with shape ``(n, 1)`` where ``n`` is the number of scores recorded.

    Args:
        tunable (btb.tuning.tunable.Tunable):
            Instance of a tunable class containing hyperparameters to be tuned.
    """

    def __init__(self, tunable):
        self.tunable = tunable
        self.trials = np.empty((0, self.tunable.K), dtype=np.float)
        self.scores = np.empty((0, 1), dtype=np.float)

    def _check_proposals(self, num_proposals):
        """Validate ``num_proposals`` with ``self.tunable.SC`` and ``self.trials``.

        Raises:
            ValueError:
                A ``ValueError`` exception is being produced if the amount of requested proposals
                is bigger than the possible combinations and ``allow_duplicates`` is ``False``.

            ValueError:
                A ``ValueError`` exception is being produced if the unique amount of recorded
                trials is the same as the amount of combinations available for ``self.tunable``.

            ValueError:
                A ``ValueError`` exception is being produced if the unique amount of recorded
                trials is the same as the amount of combinations available for ``self.tunable``.
        """
        if num_proposals > self.tunable.SC:
            raise ValueError(
                'The number of proposals requested is bigger than the combinations: {} of the'
                '``tunable``. Use ``allow_duplicates=True``, if you would like to generate that'
                'amount of combinations.'.format(self.tunable.SC)
            )

        trials_set = set(list(map(tuple, self.trials)))

        if len(trials_set) == self.tunable.SC:
            raise ValueError(
                'All of the possible combinations where recorded. Use ``allow_duplicates=True``'
                'to keep generating combinations.'
            )

        if len(trials_set) + num_proposals > self.tunable.SC:
            raise ValueError(
                'The maximum amount of new proposed combinations will exceed the amount of'
                'possible combinations, either use ``num_proposals={}`` to generate the remaining'
                'combinations or ``allow_duplicates=True`` to keep generating more'
                'combinations.'.format(self.tunable.SC - len(trials_set))
            )

    def _sample(self, num_proposals, allow_duplicates):
        """Generate a ``numpy.ndarray`` of valid proposals.

        Generates ``num_proposals`` of valid combinations by generating ``proposals`` until
        ``len(valid_proposals) == num_proposals`` different from the ones that have been recorded.

        Args:
            num_proposals (int):
                Amount of proposals to generate.
            allow_duplicates (bool):
                If it's ``False``, the tuner will propose trials that are not recorded. Otherwise
                will generate trials that may have been already recorded.

        Returns:
            numpy.ndarray:
                A ``numpy.ndarray`` with shape ``(num_proposals, self.tunable.K)``.
        """
        if allow_duplicates:
            return self.tunable.sample(num_proposals)

        else:
            valid_proposals = list()
            trials_set = set(list(map(tuple, self.trials)))

            while len(valid_proposals) < num_proposals:
                proposed = self.tunable.sample(num_proposals)
                proposed = list(map(tuple, proposed))

                if len(valid_proposals) > 0:
                    proposed = set(proposed) - set(valid_proposals)

                valid_proposals.extend(list(set(proposed) - trials_set))

            return np.asarray(valid_proposals)[:num_proposals]

    @abstractmethod
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
        pass

    def propose(self, num_proposals=1, allow_duplicates=False):
        """Propose (one or more) new hyperparameter configurations.

        Validate that the amount of proposals requested is valid when ``allow_duplicates`` is
        ``False`` and raise an exception in case there is any missmatch between ``num_proposals``,
        ``self.trials`` and ``self.tunable.SC``.
        Call the implemented ``_propose`` method and convert the returned data in to hyperparameter
        space values.

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

            ValueError:
                A ``ValueError`` exception is being produced if the unique amount of recorded
                trials is the same as the amount of combinations available for ``self.tunable``.

            ValueError:
                A ``ValueError`` exception is being produced if the unique amount of recorded
                trials is the same as the amount of combinations available for ``self.tunable``.

        Example:
            The example below shows simple usage case where an ``UniformTuner`` is being imported,
            instantiated with a ``tunable`` object and it's method propose is being called
            three times, first with a single proposal, a second with two proposals forcing them to
            be different and once where the values can be repeated.

            >>> from btb.tuning.tunable import Tunable
            >>> from btb.tuning.hyperparams import BooleanHyperParam
            >>> from btb.tuning.hyperparams import CategoricalHyperParam
            >>> from btb.tuning.tuners import UniformTuner
            >>> bhp = BooleanHyperParam()
            >>> chp = CategoricalHyperParam(['cat', 'dog'])
            >>> tunable = Tunable({'bhp': bhp, 'chp': chp})
            >>> tuner = UniformTuner(tunable)
            >>> tuner.propose(1)
            {'bhp': True, 'chp': 'dog'}
            >>> tuner.propose(2)
            [{'bhp': True, 'chp': 'cat'}, {'bhp': True, 'chp': 'dog'}]
            >>> tuner.propose(2, allow_duplicates=True)
            [{'bhp': False, 'chp': 'dog'}, {'bhp': False, 'chp': 'dog'}]
        """

        if not allow_duplicates:
            self._check_proposals(num_proposals)

        proposed = self._propose(num_proposals, allow_duplicates)
        hyperparameters = self.tunable.inverse_transform(proposed)
        hyperparameters = hyperparameters.to_dict(orient='records')

        if num_proposals == 1:
            hyperparameters = hyperparameters[0]

        return hyperparameters

    def record(self, trials, scores):
        """Record one or more ``trials`` with the associated ``scores``.

        Records one or more ``trials`` with the associated ``scores`` to it. The amount of trials
        must be equal to the amount of scores recived and vice versa.

        Args:
            trials (pandas.DataFrame, pandas.Series, dict, list(dict), 2D array-like):
                Values of shape ``(n, len(self.tunable.hyperparameters))`` or dict with keys that
                are ``self.tunable.names``.

            scores (single value or array-like):
                A single value or array-like of values representing the score achieved with the
                trials.

        Raises:
            ValueError:
                A ``ValueError`` exception is being produced if ``len(trials)`` is not equal to
                ``len(scores)``.

        Example:
            The example below shows simple usage case where an ``UniformTuner`` is being imported,
            instantiated with a ``tunable`` object and it's method record is being called two times
            with valid trials and scores.

            >>> from btb.tuning.tunable import Tunable
            >>> from btb.tuning.hyperparams import BooleanHyperParam
            >>> from btb.tuning.hyperparams import CategoricalHyperParam
            >>> from btb.tuning.tuners import UniformTuner
            >>> bhp = BooleanHyperParam()
            >>> chp = CategoricalHyperParam(['cat', 'dog'])
            >>> tunable = Tunable({'bhp': bhp, 'chp': chp})
            >>> tuner = UniformTuner(tunable)
            >>> tuner.record({'bhp': True, 'chp': 'cat'}, 0.8)
            >>> trials = [{'bhp': False, 'chp': 'cat'}, {'bhp': True, 'chp': 'dog'}]
            >>> scores = [0.8, 0.1]
            >>> tuner.record(trials, scores)
        """

        trials = self.tunable.transform(trials)
        scores = scores if isinstance(scores, (list, np.ndarray)) else [scores]

        if len(trials) != len(scores):
            raise ValueError('The amount of trials must be equal to the amount of scores.')

        self.trials = np.append(self.trials, trials, axis=0)
        self.scores = np.append(self.scores, scores)
