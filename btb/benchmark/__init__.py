# -*- coding: utf-8 -*-

import pandas as pd

from btb.benchmark.challenges.bohachevsky import Bohachevsky
from btb.benchmark.challenges.boston import BostonABR, BostonBR, BostonRFR
from btb.benchmark.challenges.branin import Branin
from btb.benchmark.challenges.census import CensusABC, CensusRFC, CensusSGDC
from btb.benchmark.challenges.rosenbrock import Rosenbrock
from btb.benchmark.challenges.wind import WindABC, WindRFC, WindSGDC

DEFAULT_CHALLENGES = [
    # Simple
    Bohachevsky,
    Branin,
    Rosenbrock,

    # ML
    BostonABR,
    BostonBR,
    BostonRFR,
    CensusABC,
    CensusRFC,
    CensusSGDC,
    WindABC,
    WindRFC,
    WindSGDC
]


def evaluate_candidate(name, candidate, challenges, iterations):
    candidate_result = []

    if not isinstance(challenges, list):
        challenges = [challenges]

    for challenge in challenges:
        tunable_hyperparameters = challenge.get_tunable_hyperparameters()
        score = candidate(challenge.evaluate, tunable_hyperparameters, iterations)
        result = {
            'challenge': str(challenge),
            'candidate': name,
            'score': score,
        }

        candidate_result.append(result)

    return candidate_result


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
        challenges = [challenge_class() for challenge_class in DEFAULT_CHALLENGES]

    if callable(candidates):
        candidates = {candidates.__name__: candidates}

    elif isinstance(candidates, (list, tuple)):
        candidates = {candidate.__name__: candidate for candidate in candidates}

    elif not isinstance(candidates, dict):
        raise TypeError(
            'Candidates can only be a callable, list of callables, tuple of callables or dict.')

    if not isinstance(challenges, list):
        challenges = [challenges]

    results = []

    for name, candidate in candidates.items():
        result = evaluate_candidate(name, candidate, challenges, iterations)
        results.extend(result)

    df = pd.DataFrame.from_records(results)
    df = df.pivot(index='candidate', columns='challenge', values='score')

    del df.columns.name
    del df.index.name

    df['Mean'] = df.mean(axis=1)
    df['Std'] = df.std(axis=1)

    return df
