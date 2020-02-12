# -*- coding: utf-8 -*-

from btb.benchmark import benchmark
from btb.benchmark.challenges import Rosenbrock
from btb.benchmark.tuners.btb import make_tuning_function
from btb.tuning import GPTuner


def test_benchmark_rosenbrock():
    candidate = make_tuning_function(GPTuner)
    benchmark(candidate, challenges=Rosenbrock(), iterations=1)

    # TODO: Add asserts
