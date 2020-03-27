# -*- coding: utf-8 -*-
import inspect
import logging

import dask
import pandas as pd

from btb.benchmark.challenges import MATH_CHALLENGES, ML_CHALLENGES

DEFAULT_CHALLENGES = MATH_CHALLENGES + ML_CHALLENGES
LOGGER = logging.getLogger(__name__)


@dask.delayed
def _evaluate_candidate(name, candidate, challenge, iterations):
    tunable_hyperparameters = challenge.get_tunable_hyperparameters()
    LOGGER.info('Evaluating candidate %s on challenge %s for %s iterations',
                name, challenge, iterations)
    try:
        score = candidate(challenge.evaluate, tunable_hyperparameters, iterations)
        result = {
            'challenge': str(challenge),
            'candidate': name,
            'score': score,
        }

    except Exception as ex:
        LOGGER.warn(
            'Could not score candidate %s with challenge %s, error: %s', name, challenge, ex)
        result = {
            'challenge': str(challenge),
            'candidate': name,
            'score': None,
        }

    return result


def evaluate_candidate(name, candidate, challenges, iterations):
    candidate_results = []
    for challenge in challenges:
        try:
            if inspect.isclass(challenge):
                challenge = challenge()

            result = _evaluate_candidate(name, candidate, challenge, iterations)

        except Exception as ex:
            LOGGER.warn(
                'Could not score candidate %s with challenge %s, error: %s', name, challenge, ex)

        candidate_results.append(result)

    return candidate_results


def benchmark(candidates, challenges=None, iterations=1000):
    """Benchmark function.

    This benchmark function iterates over a collection of ``challenges`` and executes a
    ``tuner_function`` for each one of the ``challenges`` for a given amount of iterations.

    Args:
        candidates (callable, list, tuple or dict):
            Python callable function, list of callable functions, tuple with callable functions or
            dictionary with the ``name`` of the function as ``key`` and the callable function that
            returns the best score for a given ``scorer``. This function must have three arguments:

                * scorer (function):
                    A function that performs scoring over params.

                * tunable (btb.tuning.Tunable):
                    A ``Tunable`` instance used to instantiate a tuner.

                * iterations (int):
                    Number of tuning iterations to perform.

        challenges (single challenge or list):
            A single ``challenge`` or a list of ``chalenges``. This challenges must inherit
            from ``btb.challenges.challenge.Challenge``.
        iterations (int):
            Amount of iterations to perform for the ``tuner_function``.

    Returns:
        pandas.DataFrame:
            A ``pandas.DataFrame`` with the obtained scores for the given challenges is being
            returned.
    """
    if challenges is None:
        challenges = DEFAULT_CHALLENGES
    elif not isinstance(challenges, list):
        challenges = [challenges]

    if callable(candidates):
        candidates = {candidates.__name__: candidates}
    elif isinstance(candidates, (list, tuple)):
        candidates = {candidate.__name__: candidate for candidate in candidates}
    elif not isinstance(candidates, dict):
        raise TypeError('Candidates can only be a callable, list or dict.')

    results = []

    for name, candidate in candidates.items():
        LOGGER.info('Evaluating candidate %s', name)

        result = evaluate_candidate(name, candidate, challenges, iterations)

        results.extend(result)

    results = [result for result in dask.compute(*results)]

    df = pd.DataFrame.from_records(results)
    df = df.pivot(index='challenge', columns='candidate', values='score')

    del df.columns.name
    del df.index.name

    return df
