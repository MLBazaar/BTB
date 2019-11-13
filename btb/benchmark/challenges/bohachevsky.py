# -*- coding: utf-8 -*-

import numpy as np

from btb.benchmark.challenges.challenge import Challenge
from btb.tuning import Tunable
from btb.tuning.hyperparams import IntHyperParam


class Bohachevsky(Challenge):
    """Bohachevsky challenge."""
    def __init__(self):
        pass

    @staticmethod
    def get_tunable():
        x = IntHyperParam(min=-100, max=100)
        y = IntHyperParam(min=-100, max=100)

        return Tunable({'x': x, 'y': y})

    def score(self, x, y):
        z = np.cos(3 * np.pi * x)
        return -1 * (x**2 + 2 * y**2 - 0.3 * z - 0.4 * np.cos(4 * np.pi * y) + 0.7)
