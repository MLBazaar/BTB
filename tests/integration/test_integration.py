# -*- coding: utf-8 -*-

import random

from btb.benchmark import benchmark
from btb.benchmark.challenges import Rosenbrock
from btb.benchmark.tuners.btb import make_tuning_function
from btb.tuning import GPEiTuner, GPTuner, Tunable
from btb.tuning.hyperparams import (
    BooleanHyperParam, CategoricalHyperParam, FloatHyperParam, IntHyperParam)


def test_benchmark_rosenbrock():
    candidate = make_tuning_function(GPTuner)
    benchmark(candidate, challenges=Rosenbrock(), iterations=1)


def test_tunable_tuner():

    hyperparams = {
        'bhp': BooleanHyperParam(default=False),
        'chp': CategoricalHyperParam(choices=['a', 'b', None], default=None),
        'fhp': FloatHyperParam(min=0.1, max=1.0, default=0.5),
        'ihp': IntHyperParam(min=-1, max=1)
    }

    tunable = Tunable(hyperparams)

    tuner = GPEiTuner(tunable)

    for _ in range(10):
        proposed = tuner.propose(1)
        tuner.record(proposed, random.random())
