# -*- coding: utf-8 -*-

import random

from btb.tuning import GPTuner, Tunable
from btb.tuning.hyperparams import (
    BooleanHyperParam, CategoricalHyperParam, FloatHyperParam, IntHyperParam)


def test_tuning():
    hyperparams = {
        'bhp': BooleanHyperParam(default=False),
        'chp': CategoricalHyperParam(choices=['a', 'b', None], default=None),
        'fhp': FloatHyperParam(min=0.1, max=1.0, default=0.5),
        'ihp': IntHyperParam(min=-1, max=1)
    }
    tunable = Tunable(hyperparams)
    tuner = GPTuner(tunable)

    for _ in range(10):
        proposed = tuner.propose(1)
        tuner.record(proposed, random.random())

    # TODO add asserts
