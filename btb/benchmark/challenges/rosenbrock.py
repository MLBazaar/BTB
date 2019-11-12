# -*- coding: utf-8 -*-

from btb.benchmark.challenges.challenge import Challenge
from btb.tuning import Tunable
from btb.tuning.hyperparams import IntHyperParam


class Rosenbrock(Challenge):
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    @staticmethod
    def get_tunable():
        x = IntHyperParam(min=-50, max=50)
        y = IntHyperParam(min=-50, max=50)

        return Tunable({'x': x, 'y': y})

    def score(self, x, y):
        return -1 * ((self.a - x)**2 + self.b * (y - x**2)**2)
