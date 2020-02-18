# -*- coding: utf-8 -*-
import numpy as np

from btb.benchmark import benchmark
from btb.benchmark.challenges import Rosenbrock
from btb.benchmark.tuners.btb import make_tuning_function
from btb.tuning import GPTuner


def test_benchmark_rosenbrock():
    # run
    candidate = make_tuning_function(GPTuner)
    df = benchmark(candidate, challenges=Rosenbrock(), iterations=1)

    # Assert
    np.testing.assert_equal(df.columns.values, ['Rosenbrock()', 'Mean', 'Std'])
    np.testing.assert_equal(df.index.values, ['tuning_function'])
    np.testing.assert_equal(df.dtypes.values, [np.int, np.float, np.float])
