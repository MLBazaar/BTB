# -*- coding: utf-8 -*-

import numpy as np

from btb.tuning.tunable import Tunable
from btb.tuning.tuners import GPEiTuner, GPTuner, UniformTuner


def tuning_function(tuner_class, scoring_function, tunable_hyperparameters, iterations):
    tunable = Tunable.from_dict(tunable_hyperparameters)
    tuner = tuner_class(tunable)
    best_score = -np.inf

    for _ in range(iterations):
        proposal = tuner.propose()
        score = scoring_function(**proposal)
        tuner.record(proposal, score)
        best_score = max(score, best_score)

    return best_score


def gptuner_tuning_function(scoring_function, tunable_hyperparameters, iterations):
    return tuning_function(GPTuner, scoring_function, tunable_hyperparameters, iterations)


def gpeituner_tuning_function(scoring_function, tunable_hyperparameters, iterations):
    return tuning_function(GPEiTuner, scoring_function, tunable_hyperparameters, iterations)


def uniformtuner_tuning_function(scoring_function, tunable_hyperparameters, iterations):
    return tuning_function(UniformTuner, scoring_function, tunable_hyperparameters, iterations)
