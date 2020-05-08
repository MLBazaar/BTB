# -*- coding: utf-8 -*-

"""Top level where all the challenges are imported."""

from btb_benchmark.challenges.bohachevsky import Bohachevsky
from btb_benchmark.challenges.branin import Branin
from btb_benchmark.challenges.randomforest import RandomForestChallenge
from btb_benchmark.challenges.rosenbrock import Rosenbrock
from btb_benchmark.challenges.sgd import SGDChallenge
from btb_benchmark.challenges.xgboost import XGBoostChallenge

__all__ = (
    'Bohachevsky',
    'Branin',
    'RandomForestChallenge',
    'Rosenbrock',
    'SGDChallenge',
    'XGBoostChallenge',
)

MATH_CHALLENGES = {
    'bohachevsky': Bohachevsky,
    'branin': Branin,
    'rosenbrock': Rosenbrock,
}
