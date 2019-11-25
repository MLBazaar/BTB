# -*- coding: utf-8 -*-

import pandas as pd

from btb.benchmark.challenges import Bohachevsky, Branin, Rosenbrock

DEFAULT_CHALLENGES = [
    Bohachevsky,
    Branin,
    Rosenbrock,
]


def benchmark(candidates, challenges=DEFAULT_CHALLENGES, iterations=1000):
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

    for challenge_class in challenges:
        challenge = challenge_class()
        tunable = challenge.get_tunable()

        for name, function in candidates.items():
            score = function(challenge.evaluate, tunable, iterations)

            results.append({
                'challenge': type(challenge).__name__,
                'tuner': name,
                'score': score,
            })

    df = pd.DataFrame.from_records(results)
    df = df.pivot(index='tuner', columns='challenge', values='score')

    del df.columns.name
    del df.index.name

    df['Mean'] = df.mean(axis=1)
    df['Std'] = df.std(axis=1)

    return df
