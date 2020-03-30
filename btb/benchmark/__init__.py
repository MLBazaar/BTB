# -*- coding: utf-8 -*-
import logging
from datetime import datetime

import dask
import pandas as pd

from btb.benchmark.challenges import MATH_CHALLENGES, Challenge

LOGGER = logging.getLogger(__name__)


@dask.delayed
def _evaluate_tuner(name, tuner, challenge, iterations):
    tunable_hyperparameters = challenge.get_tunable_hyperparameters()
    LOGGER.info('Evaluating tuner %s on challenge %s for %s iterations',
                name, challenge, iterations)
    try:
        start = datetime.utcnow()
        score = tuner(challenge.evaluate, tunable_hyperparameters, iterations)
        result = {
            'challenge': str(challenge),
            'tuner': name,
            'score': score,
            'elapsed': datetime.utcnow() - start
        }

    except Exception as ex:
        LOGGER.warn(
            'Could not score tuner %s with challenge %s, error: %s', name, challenge, ex)
        result = {
            'challenge': str(challenge),
            'tuner': name,
            'score': None,
            'elapsed': datetime.utcnow() - start
        }

    return result


def evaluate_tuner(name, tuner, challenges, iterations):
    tuner_results = []
    for challenge in challenges:
        try:
            if not isinstance(challenge, Challenge) and issubclass(challenge, Challenge):
                challenge = challenge()

            result = _evaluate_tuner(name, tuner, challenge, iterations)

        except Exception as ex:
            LOGGER.warn(
                'Could not score tuner %s with challenge %s, error: %s', name, challenge, ex)

        tuner_results.append(result)

    return tuner_results


def benchmark(tuners, challenges=None, iterations=1000, verbose_dataframe=False):
    """Benchmark function.

    This benchmark function iterates over a collection of ``challenges`` and executes a
    ``tuner_function`` for each one of the ``challenges`` for a given amount of iterations.

    Args:
        tuners (callable, list, tuple or dict):
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
        challenges = MATH_CHALLENGES
    elif not isinstance(challenges, list):
        challenges = [challenges]

    if callable(tuners):
        tuners = {tuners.__name__: tuners}
    elif isinstance(tuners, (list, tuple)):
        tuners = {tuner.__name__: tuner for tuner in tuners}
    elif not isinstance(tuners, dict):
        raise TypeError('tuners can only be a callable, list or dict.')

    results = []

    for name, tuner in tuners.items():
        LOGGER.info('Evaluating tuner %s', name)

        result = evaluate_tuner(name, tuner, challenges, iterations)

        results.extend(result)

    results = [result for result in dask.compute(*results)]

    df = pd.DataFrame.from_records(results)

    if verbose_dataframe:
        return df

    df = df.pivot(index='challenge', columns='tuner', values='score')

    del df.columns.name
    del df.index.name

    return df
