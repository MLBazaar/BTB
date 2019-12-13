# -*- coding: utf-8 -*-

import json
import logging
from collections import Counter, defaultdict
from hashlib import md5

import numpy as np
from tqdm.autonotebook import trange

from btb.selection.ucb1 import UCB1
from btb.tuning.tunable import Tunable
from btb.tuning.tuners.gaussian_process import GPTuner

LOGGER = logging.getLogger(__name__)


class BTBSession:
    """BTBSession class.

    BTBSession class creates an object that tunes a list of ``tunables`` using a
    ``tuner`` and a ``selector`` in order to determinate the best ``tunable`` and the
    best ``configuration`` for it.

    Args:
        tunables (dict):
            Python dictionary that has as keys the name of the tunable and
            as value a dictionary with the tunable hyperparameters.
        scorer (callable object / function):
            A scorer callable object or function with signature ``scorer(tunable_name, config)``
            wich should return only a single value.
        tuner (btb.tuning.tuner.BaseTuner):
            A tuner based on BTB ``BaseTuner`` class. This tuner will manage the new proposals.
            Defaults to ``btb.tuning.tuners.gaussian_process.GPTuner``
        selector (btb.selection.selector.Selector):
            A selector based on BTB ``Selector`` class. This will determinate which one of
            the tunables is performing better. Defaults to ``btb.selection.selectors.ucb1.UCB1``
        maximize (bool):
            If ``True`` the scores are interpreted as bigger is better, if ``False`` then smaller
            is better. Defaults to ``True``.
        max_erors (int):
            Amount of errors allowed for a tunable to not generate a score.
        verbose (bool):
            If ``True`` a progress bar will be displayed for the ``run`` process.
    """
    tunables = None
    scorer = None
    tuner = None
    selector = None
    maximize = None
    max_errors = None

    best_proposal = None
    proposals = None
    iterations = None
    errors = None

    _best_normalized = None
    _tunable_names = None
    _normalized_scores = None
    _tuners = None

    def _normalize(self, score):
        if score is not None:
            return score if self.maximize else -score

    def __init__(self, tunables, scorer, tuner=GPTuner, selector=UCB1,
                 maximize=True, max_errors=1, verbose=False):

        self.tunables = tunables
        self.scorer = scorer
        self.tuner = tuner
        self.max_errors = max_errors
        self.maximize = maximize

        self.best_proposal = None
        self.proposals = dict()
        self.iterations = 0
        self.errors = Counter()

        self._best_normalized = self._normalize(-np.inf)
        self._normalized_scores = defaultdict(list)
        self._tuners = dict()
        self._tunable_names = list(self.tunables.keys())
        self.selector = selector(self._tunable_names)
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
        """Propose a new configuration for a tunable.

        By accessing the dictionary ``self.tunables``, ``BTBSession``, ensures that
        every tunable has been scored atleast once. The following proposals use the
        ``self.selector`` in order to select the ``tunable`` from which a proposal
        is generated.

        In case that the ``tuner`` can't propose more configurations it will return
        ``None`` and will remove the ``tunable`` from the list.

        Returns:
            tupple (str, dict):
                Returns a tupple with the name of the tunable and the proposal as a dictionary.
            None:
                ``None`` is being returned When the ``tunable`` has no more combinations to be
                evaluated.
        """
        if not self.tunables:
            raise ValueError('All the tunables failed.')

        if len(self._normalized_scores) < len(self._tunable_names):
            tunable_name = self._tunable_names[len(self._normalized_scores)]
            tunable_spec = self.tunables[tunable_name]

            tunable = Tunable.from_dict(tunable_spec)
            LOGGER.info('Obtaining default configuration for %s', tunable_name)
            config = tunable.get_defaults()

            self._tuners[tunable_name] = self.tuner(tunable)

        else:
            tunable_name = self.selector.select(self._normalized_scores)
            tuner = self._tuners[tunable_name]
            try:
                LOGGER.info('Generating new proposal configuration for %s', tunable_name)
                config = tuner.propose(1)

            except ValueError:
                LOGGER.info('%s has no more configs to propose.' % tunable_name)
                self._normalized_scores.pop(tunable_name, None)
                self._tunable_names.remove(tunable_name)
                return None

        proposal_id = self._make_id(tunable_name, config)
        self.proposals[proposal_id] = {
            'id': proposal_id,
            'name': tunable_name,
            'config': config
        }

        return tunable_name, config

    def record(self, tunable_name, config, score):
        """Record the configuration and the associated score to it inside the tuner.

        Records the associated configuration and score to the tuner. Evaluates if the
        score is the best score found, if so, updates ``self._best_normalized`` and
        ``self.best_proposal`` with the associated to them values.

        Args:
            tunable_name (str):
                The name of the tunable that the score and config pertain.
            config (dict):
                Dictionary representation of the configuration given to the tunable to obtain
                the score.
            score (float):
                Obtained score with the given configuration.
        """
        proposal_id = self._make_id(tunable_name, config)
        proposal = self.proposals[proposal_id]
        proposal['score'] = score

        if score is None:
            self.errors[tunable_name] += 1
            errors = self.errors[tunable_name]

            if errors >= self.max_errors:
                LOGGER.warning('Too many errors: %s. Removing tunable %s', errors, tunable_name)
                self._normalized_scores.pop(tunable_name, None)
                self._tunable_names.remove(tunable_name)
        else:
            normalized = self._normalize(score)
            if normalized > self._best_normalized:
                LOGGER.info('New optimal found: %s - %s', tunable_name, score)
                self.best_proposal = proposal
                self._best_normalized = normalized
            try:
                tuner = self._tuners[tunable_name]
                tuner.record(config, normalized)
                self._normalized_scores[tunable_name].append(normalized)
            except Exception:
                LOGGER.exception('Could not record score to tuner.')

    def run(self, iterations=10):
        """Execute proposal iterations over the tunables.

        Given a number of ``iterations`` propose new configurations for the tunables,
        score and record those inside the ``tuner`` object.

        Returns:
            best_proposal (dict):
                Best configuration found with the name of the tunable and the hyperparameters
                and crossvalidated score obtained for it.
        """
        for _ in self._range(iterations):
            self.iterations += 1
            proposal = self.propose()
            if proposal is None:
                continue

            tunable_name, config = proposal

            try:
                LOGGER.debug('Scoring proposal %s - %s: %s', self.iterations, tunable_name, config)
                score = self.scorer(tunable_name, config)

            except Exception:
                LOGGER.exception('Proposal %s - %s crashed', self.iterations, tunable_name)
                score = None

            self.record(tunable_name, config, score)

        return self.best_proposal
