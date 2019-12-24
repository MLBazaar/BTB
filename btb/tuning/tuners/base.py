# -*- coding: utf-8 -*-

"""Package where the BaseTuner class and BaseMetaModelTuner are defined."""

import logging
from abc import abstractmethod

import numpy as np

from btb.tuning.acquisition.base import BaseAcquisition
from btb.tuning.metamodels.base import BaseMetaModel

LOGGER = logging.getLogger(__name__)


class StopTuning(Exception):
    pass


class BaseTuner:
    """BaseTuner class.

    BaseTuner class is the abstract representation of a tuner that is not based on a model.

    Attributes:
        tunable (btb.tuning.tunable.Tunable):
            Instance of a tunable class containing hyperparameters to be tuned.
        trials (numpy.ndarray):
            A ``numpy.ndarray`` with shape ``(n, self.tunable.dimensions)`` where ``n`` is the
            number of trials recorded.
        scores (numpy.ndarray):
            A ``numpy.ndarray`` with shape ``(n, 1)`` where ``n`` is the number of scores recorded.

    Args:
        tunable (btb.tuning.tunable.Tunable):
            Instance of a tunable class containing hyperparameters to be tuned.
        maximize (bool):
            If ``True`` the scores are interpreted as bigger is better, if ``False`` then smaller
            is better. Defaults to ``True``.
    """

    def __init__(self, tunable, maximize=True):
        self.tunable = tunable
        self.trials = np.empty((0, self.tunable.dimensions), dtype=np.float)
        self._trials_set = set()
        self.raw_scores = np.empty((0, 1), dtype=np.float)
        self.maximize = maximize
        LOGGER.debug(
            ('Creating %s instance with %s hyperparameters and carinality %s.'),
            len(self.tunable.hyperparams), self.__class__.__name__, self.tunable.cardinality
        )

    def _check_proposals(self, num_proposals):
        """Validate ``num_proposals`` with ``self.tunable.cardinality`` and ``self.trials``.

        Raises:
            StopTuning:
                A ``StopTuning`` exception is being produced if the amount of requested proposals
                is bigger than the possible combinations and ``allow_duplicates`` is ``False``.
            StopTuning:
                A ``StopTuning`` exception is being produced if the unique amount of recorded
                trials is the same as the amount of combinations available for ``self.tunable``.
            StopTuning:
                A ``StopTuning`` exception is being produced if the unique amount of recorded
                trials is the same as the amount of combinations available for ``self.tunable``.
        """
        if num_proposals > self.tunable.cardinality:
            raise StopTuning(
                'The number of proposals requested is bigger than the combinations: {} of the'
                '``tunable``. Use ``allow_duplicates=True``, if you would like to generate that'
                'amount of combinations.'.format(self.tunable.cardinality)
            )

        num_tried = len(self._trials_set)
        if num_tried == self.tunable.cardinality:
            raise StopTuning(
                'All of the possible combinations where recorded. Use ``allow_duplicates=True``'
                'to keep generating combinations.'
            )

        if num_tried + num_proposals > self.tunable.cardinality:
            raise StopTuning(
                'The maximum amount of new proposed combinations will exceed the amount of'
                'possible combinations, either use ``num_proposals={}`` to generate the remaining'
                'combinations or ``allow_duplicates=True`` to keep generating more'
                'combinations.'.format(self.tunable.cardinality - num_tried)
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
                A ``numpy.ndarray`` with shape ``(num_proposals, self.tunable.dimensions)``.
        """
        if allow_duplicates:
            return self.tunable.sample(num_proposals)

        else:
            valid_proposals = set()

            while len(valid_proposals) < num_proposals:
                proposals = self.tunable.sample(num_proposals)
                proposals = set(map(tuple, proposals))

                valid_proposals.update(proposals - self._trials_set)

            return np.asarray(list(valid_proposals))[:num_proposals]

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

    def propose(self, n=1, allow_duplicates=False):
        """Propose one or more new hyperparameter configurations.

        Validate that the amount of proposals requested is valid when ``allow_duplicates`` is
        ``False`` and raise an exception in case there is any missmatch between ``n``,
        unique ``self.trials`` and ``self.tunable.cardinality``.
        Call the implemented ``_propose`` method and convert the returned data in to hyperparameter
        space values.

        Args:
            n (int):
                Number of candidates to create. Defaults to 1.
            allow_duplicates (bool):
                If it's False, the tuner will propose trials that are not recorded. Otherwise
                will generate trials that can be repeated. Defaults to ``False``.

        Returns:
            dict or list:
                If ``n`` is 1, a ``dict`` will be returned containing the
                hyperparameter names and values. Otherwise, if ``n`` is bigger than 1,
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
            self._check_proposals(n)

        proposed = self._propose(n, allow_duplicates)

        hyperparameters = self.tunable.inverse_transform(proposed)
        hyperparameters = hyperparameters.to_dict(orient='records')

        if n == 1:
            hyperparameters = hyperparameters[0]

        return hyperparameters

    def record(self, trials, scores):
        """Record one or more ``trials`` with the associated ``scores``.

        ``Trials`` are recorded with their associated ``scores``. The amount of trials
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
        self._trials_set.update(map(tuple, trials))
        self.raw_scores = np.append(self.raw_scores, scores)
        self.scores = self.raw_scores if self.maximize else -self.raw_scores

    def __str__(self):
        return (
            "{}\n"
            "  hyperparameters: {}\n"
            "  dimensions: {}\n"
            "  cardinality: {}"
        ).format(
            self.__class__.__name__,
            len(self.tunable.hyperparams),
            self.tunable.dimensions,
            self.tunable.cardinality
        )


class BaseMetaModelTuner(BaseTuner, BaseMetaModel, BaseAcquisition):
    """BaseMetaModelTuner class.

    BaseMetaModelTuner class is the abstract representation of a tuner that is based
    on a model and an ``Acquisition``. This model will try to `predict` the
    score that will be obtained with the proposed parameters by being trained
    over the ``self.trials`` and ``self.raw_scores`` recorded by the user.

    Attributes:
        tunable (btb.tuning.tunable.Tunable):
            Instance of a tunable class containing hyperparameters to be tuned.
        trials (numpy.ndarray):
            A ``numpy.ndarray`` with shape ``(n, self.tunable.dimensions)`` where ``n`` is the
            number of trials recorded.
        scores (numpy.ndarray):
            A ``numpy.ndarray`` with shape ``(n, 1)`` where ``n`` is the number of scores recorded.

    Args:
        tunable (btb.tuning.tunable.Tunable):
            Instance of a tunable class containing hyperparameters to be tuned.
        num_candidates (int):
            Number of samples to generate and select the best of it for each proposal. Defaults to
            1000.
        maximize (bool):
            If ``True`` the model will understand that the score bigger is better, if ``False``
            the smaller is better. Defaults to ``True``.
        min_trials (int):
            Number of recorded ``trials`` needed to perform a fitting over the model.
            Defaults to 2.
    """

    _metamodel_kwargs = None
    _acquisition_kwargs = None

    def __init__(self, tunable, maximize=True, num_candidates=1000, min_trials=2):
        self.num_candidates = num_candidates
        self.min_trials = min_trials
        super().__init__(tunable, maximize)
        self.__init_metamodel__(**(self._metamodel_kwargs or dict()))
        self.__init_acquisition__(**(self._acquisition_kwargs or dict()))

    def _propose(self, num_proposals, allow_duplicates):
        if self.min_trials > len(self._trials_set):
            LOGGER.debug('Not enough samples recorded to generate predictions, '
                         'generating random proposal.')
            return self._sample(num_proposals, allow_duplicates)

        num_samples = num_proposals * self.num_candidates
        if not allow_duplicates:
            remaining = self.tunable.cardinality - len(self._trials_set)
            num_samples = min(remaining, num_samples)

        proposals = self._sample(num_samples, allow_duplicates)

        predicted = self._predict(proposals)
        index = self._acquire(predicted, num_proposals)

        return proposals[index]

    def record(self, trials, scores):
        """Record one or more ``trials`` with the associated ``scores`` and re-fit the model.

        ``Trials`` are recorded with the associated ``scores`` to them. The amount of trials
        must be equal to the amount of scores recived and vice versa. Once recorded, the ``model``
        is being fitted with ``self.trials`` and ``self.raw_scores`` that contain any previous
        records and the ones that where just recorded.

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
        super().record(trials, scores)
        if len(self.trials) >= self.min_trials:
            LOGGER.debug('Fitting the model with %s samples.' % len(self.trials))
            self._fit(self.trials, self.scores)
