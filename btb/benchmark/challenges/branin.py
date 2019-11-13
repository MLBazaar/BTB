# -*- coding: utf-8 -*-

import numpy as np

from btb.benchmark.challenges.challenge import Challenge
from btb.tuning import Tunable
from btb.tuning.hyperparams import FloatHyperParam


class Branin(Challenge):
    """Branin challenge."""
    def __init__(self, a=1, b=5.1, c=5, r=6, s=10, t=1):
        self.a = a
        self.b = b / (4 * pow(np.pi, 2))
        self.c = c / np.pi
        self.r = r
        self.s = s
        self.t = t / (8 * np.pi)

    @staticmethod
    def get_tunable():
        x = FloatHyperParam(min=-5.0, max=10.0)
        y = FloatHyperParam(min=0.0, max=15.0)

        return Tunable({'x': x, 'y': y})

    def score(self, x, y):
        z = (y - self.b * x**2 + self.c * x - self.r)**2
        return -1 * (self.a * z + self.s * (1 - self.t) * np.cos(x) + self.s)
