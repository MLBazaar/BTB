# -*- coding: utf-8 -*-

import pandas as pd

from btb.challenges import Rosenbrock

DEFAULT_CHALLENGES = [
    Rosenbrock,
]


def benchmark(tuner_function, challenges=DEFAULT_CHALLENGES, iterations=1000):
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
