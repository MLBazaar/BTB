# -*- coding: utf-8 -*-

import numpy as np

from btb.tuning.tunable import Tunable


def tune(tuner, scoring_function, iterations):
    best_score = -np.inf

    for _ in range(iterations):
        proposal = tuner.propose()
        score = scoring_function(**proposal)
        tuner.record(proposal, score)
        best_score = max(score, best_score)

    return best_score


def make_tuning_function(tuner_class, **tuner_kwargs):
    def tuning_function(scoring_function, tunable_hyperparameters, iterations):
        tunable = Tunable.from_dict(tunable_hyperparameters)
        tuner = tuner_class(tunable, **tuner_kwargs)
        return tune(tuner, scoring_function, iterations)

    return tuning_function
