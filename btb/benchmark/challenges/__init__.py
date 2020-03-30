# -*- coding: utf-8 -*-

"""Top level where all the challenges are imported."""

from btb.benchmark.challenges.bohachevsky import Bohachevsky
from btb.benchmark.challenges.branin import Branin
from btb.benchmark.challenges.challenge import Challenge
from btb.benchmark.challenges.randomforest import RandomForestChallenge
from btb.benchmark.challenges.rosenbrock import Rosenbrock

__all__ = (
    'Bohachevsky',
    'Branin',
    'Challenge',
    'RandomForestChallenge',
    'Rosenbrock',
)

MATH_CHALLENGES = [Bohachevsky, Branin, Rosenbrock]
