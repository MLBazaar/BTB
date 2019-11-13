# -*- coding: utf-8 -*-

"""Package where the benchmark method is defined."""

import pandas as pd

from btb.benchmark.challenges import Bohachevsky, Branin, Rosenbrock

DEFAULT_CHALLENGES = [
    Bohachevsky,
    Branin,
    Rosenbrock,
]


def benchmark(tuner_function, challenges=DEFAULT_CHALLENGES, iterations=1000):
    """Benchmark function.

    This benchmark function iterates over a collection of ``challenges`` and executes a
    ``tuner_function`` for each one of the ``challenges`` for a given amount of iterations.

    Args:
        tuner_function (function):
            Python function that returns the best score for a given ``scorer``. This function
            must have three arguments:

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

    if not isinstance(challenges, list):
        challenges = [challenges]

    results = list()

    for challenge_class in challenges:
        challenge = challenge_class()
        tunable = challenge.get_tunable()

        score = tuner_function(challenge.score, tunable, iterations)

        result = pd.Series({
            'score': score,
            'iterations': iterations,
            'challenge': challenge_class.__name__,
        })

        results.append(result)

    return pd.DataFrame(results)
