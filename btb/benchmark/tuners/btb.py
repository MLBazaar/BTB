# -*- coding: utf-8 -*-

import numpy as np


def tune(tuner, scoring_function, iterations):
    best_score = -np.inf

    for _ in range(iterations):
        proposal = tuner.propose()
        score = scoring_function(**proposal)
        tuner.record(proposal, score)
        best_score = max(score, best_score)

    return best_score


def make_tuning_function(tuner_class):
    def tuning_function(scoring_function, tunable, iterations, **tuner_kwargs):
        tuner = tuner_class(tunable, **tuner_kwargs)
        return tune(tuner, scoring_function, iterations)

    return tuning_function
