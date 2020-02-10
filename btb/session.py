# -*- coding: utf-8 -*-

import itertools
import json
import logging
from collections import Counter, defaultdict
from hashlib import md5

import numpy as np
from tqdm.autonotebook import trange

from btb.selection.ucb1 import UCB1
from btb.tuning.tunable import Tunable
from btb.tuning.tuners.base import StopTuning
from btb.tuning.tuners.gaussian_process import GPTuner

LOGGER = logging.getLogger(__name__)


class BTBSession:
    """BTBSession class.

    A BTBSession represents the process of selecting and tuning several tunables
    until the best possible configuration for a specific ``scorer`` is found.

    For this, a loop is run in which for each iteration a combination of a ``Selector`` and
    ``Tuner`` is used to decide which tunable to score next and with which hyperparameters.

    While running, the ``BTBSession`` handles the errors discarding, if configured to do so,
    the tunables that have reached as many errors as the user specified.

    Attributes:
        _tunables (dict):
            Python dictionary that has as keys the name of the tunable and
            as value a dictionary with the tunable hyperparameters or an
            ``btb.tuning.tunable.Tunable`` instance.
        _scorer (callable object / function):
            A callable object or function with signature ``scorer(tunable_name, config)``
            wich should return only a single value.
        _tuner_class (btb.tuning.tuner.BaseTuner):
            A tuner based on BTB ``BaseTuner`` class. This tuner will manage the new proposals.
            Defaults to ``btb.tuning.tuners.gaussian_process.GPTuner``
        _selector_class (btb.selection.selector.Selector):
            A selector based on BTB ``Selector`` class. This will determinate which one of
            the tunables is performing better, and which one to test next. Defaults to
            ``btb.selection.selectors.ucb1.UCB1``
        _maximize (bool):
            If ``True`` the scores are interpreted as bigger is better, if ``False`` then smaller
            is better, this should depend on the problem type (maximization or minimization).
            Defaults to ``True``.
        _max_erors (int):
            Amount of errors allowed for a tunable to not generate a score. Once this amount of
            errors is reached, the tunable will be removed from the list. Defaults to 1.
        best_proposal (dict):
            Best configuration found with the name of the tunable and the hyperparameters
            and crossvalidated score obtained for it.
        best_score (float):
            Best score obtained for this session so far.
        proposals (dict):
            Dictionary containing all the proposals generated by the ``BTBSession``.
        iterations (int):
            Amount of iterations run.
        errors (list):
            A list with produced errors during the session.
        _best_normalized (float):
            Best normalized score obtained.
        _tunable_names (list):
            A list that contains the tunables that still have proposals.
        _normalized_scores (defaultdict):
            Dictionary with the name of the tunables and the obtained normalized scores.
        _tuners (dict):
            The name of the tunable and the tuner instance to which this belongs.
        verbose (bool):
            If ``True`` a progress bar will be displayed for the ``run`` process.
    """
    _tunables = None
    _scorer = None
    _tuner_class = None
    _selector = None
    _maximize = None
    _max_errors = None

    best_proposal = None
    best_score = None
    proposals = None
    iterations = None
    errors = None

    _best_normalized = None
    _tunable_names = None
    _normalized_scores = None
    _tuners = None
    _range = None

    def _normalize(self, score):
        if score is not None:
            return score if self._maximize else -score

    def __init__(self, tunables, scorer, tuner_class=GPTuner, selector_class=UCB1,
                 maximize=True, max_errors=1, verbose=False):

        self._tunables = tunables
        self._scorer = scorer
        self._tuner_class = tuner_class
        self._tunable_names = list(self._tunables.keys())
        self._selector = selector_class(self._tunable_names)
        self._maximize = maximize
        self._max_errors = max_errors

        self.best_proposal = None
        self.proposals = dict()
        self.iterations = 0
        self.errors = Counter()
        self.best_score = None

        self._best_normalized = self._normalize(-np.inf)
        self._normalized_scores = defaultdict(list)
        self._tuners = dict()
        self._range = trange if verbose else range

    def _make_dumpable(self, to_dump):
        dumpable = {}
        for key, value in to_dump.items():
            if not isinstance(key, str):
                key = str(key)

            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.bool_):
                value = bool(value)
            elif value == 'None':
                value = None

            dumpable[key] = value

        return dumpable

    def _make_id(self, name, config):
        dumpable_config = self._make_dumpable(config)
        proposal = {
            'name': name,
            'config': dumpable_config,
        }
        hashable = json.dumps(proposal, sort_keys=True).encode()

        return md5(hashable).hexdigest()

    def propose(self):
        """Propose a new configuration to score.

        Every time ``propose`` is called, a new tunable will be selected and a new
        hyperparameter proposal will be generated for it.

        At the begining, the default hyperparameters of each one of the tunables
        will be returned sequencially in the same order as they were passed to
        the ``BTBSession``.

        After that, once each tunable has been scored at least once, the tunable
        used to generate the new proposals will be selected optimally each time
        by the selector.

        If a tunable runs out of proposals, it will be discarded from the list and will
        not be proposed again.

        Finally, when all the tunables have ran out of proposals, a ``StopTuning`` exception
        will be raised.

        Returns:
            tuple (str, dict):
                * Name of the tunable to try next.
                * Hyperparameters proposal.

        Raises:
            StopTuning:
                If the ``BTBSession`` has run out of proposals to generate.
        """
        if not self._tunable_names:
            raise StopTuning('There are no tunables left to try.')

        if len(self._tuners) < len(self._tunable_names):
            tunable_name = self._tunable_names[len(self._normalized_scores)]
            tunable = self._tunables[tunable_name]

            if isinstance(tunable, dict):
                LOGGER.info('Creating Tunable instance from dict.')
                tunable = Tunable.from_dict(tunable)

            if not isinstance(tunable, Tunable):
                raise TypeError('Tunable can only be an instance of btb.tuning.Tunable or dict')

            LOGGER.info('Obtaining default configuration for %s', tunable_name)
            config = tunable.get_defaults()

            self._tuners[tunable_name] = self._tuner_class(tunable)

        else:
            tunable_name = self._selector.select(self._normalized_scores)
            tuner = self._tuners[tunable_name]
            try:
                LOGGER.info('Generating new proposal configuration for %s', tunable_name)
                config = tuner.propose(1)

            except StopTuning:
                LOGGER.info('%s has no more configs to propose.' % tunable_name)
                self._normalized_scores.pop(tunable_name, None)
                self._tunable_names.remove(tunable_name)
                tunable_name, config = self.propose()

        proposal_id = self._make_id(tunable_name, config)
        self.proposals[proposal_id] = {
            'id': proposal_id,
            'name': tunable_name,
            'config': config
        }

        return tunable_name, config

    def handle_error(self, tunable_name):
        """Handle errors when ``score`` is ``None``.

        If the given ``tunable_name`` accumulates more errors than ``self._max_errors``
        this is removed from the selector's choices.

        Args:
            tunable_name (str):
                The name of the tunable to which this configuration belongs.
        """
        self.errors[tunable_name] += 1
        errors = self.errors[tunable_name]

        if errors >= self._max_errors:
            LOGGER.warning('Too many errors: %s. Removing tunable %s', errors, tunable_name)
            self._normalized_scores.pop(tunable_name, None)
            self._tunable_names.remove(tunable_name)

    def record(self, tunable_name, config, score):
        """Record the configuration and the obtained score to the tuner.

        If the score is the best one so far, the ``best_proposal`` and ``best_score`` are
        updated.

        Args:
            tunable_name (str):
                The name of the tunable to which this configuration belongs.
            config (dict):
                Hyperparameter proposal, as given by the tunable.
            score (float):
                Obtained score with the given configuration.
        """
        proposal_id = self._make_id(tunable_name, config)
        proposal = self.proposals[proposal_id]
        proposal['score'] = score

        if score is None:
            self.handle_error(tunable_name)
        else:
            normalized = self._normalize(score)
            self._normalized_scores[tunable_name].append(normalized)

            if normalized > self._best_normalized:
                LOGGER.info('New optimal found: %s - %s', tunable_name, score)
                self.best_proposal = proposal
                self.best_score = score
                self._best_normalized = normalized
            try:
                tuner = self._tuners[tunable_name]
                tuner.record(config, normalized)
            except Exception:
                LOGGER.exception('Could not record configuration and score to tuner.')

    def run(self, iterations=None):
        """Run the selection and tuning loop for the given number of iterations.

        At each iteration, the `BTBSession` will generate a new proposal calling
        ``self.propose``, score it using the `self.scorer`, and finally record the
        obtained score back to the tuner calling `self.record`.

        If no iterations are given, run infinitely until interrupted or until all the
        tuner proposals are exhausted.

        Scoring errors will also be captured and recorded.

        Returns:
            best_proposal (dict):
                Best configuration found with the name of the tunable and the hyperparameters
                and crossvalidated score obtained for it.
        """
        if iterations is None:
            iterator = itertools.count()
        else:
            iterator = self._range(iterations)

        for _ in iterator:
            self.iterations += 1
            tunable_name, config = self.propose()

            try:
                LOGGER.debug('Scoring proposal %s - %s: %s', self.iterations, tunable_name, config)
                score = self._scorer(tunable_name, config)

            except Exception:
                params = '\n'.join('{}: {}'.format(k, v) for k, v in config.items())
                LOGGER.exception(
                    'Proposal %s - %s crashed with the following configuration: %s',
                    self.iterations,
                    tunable_name,
                    params
                )

                score = None

            self.record(tunable_name, config, score)

        return self.best_proposal
