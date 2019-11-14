# -*- coding: utf-8 -*-

import numpy as np

from btb.benchmark.challenges.challenge import Challenge
from btb.tuning import Tunable
from btb.tuning.hyperparams import IntHyperParam


class Bohachevsky(Challenge):
    r"""Bohachevsky challenge.

    The Bohachevsky functions are bowl shape functions. This function is usually evaluated on the
    square :math:`xii \Epsilon [-100, 100]` square for all :math:`i = 1, 2`. For more information
    please visit: https://www.sfu.ca/~ssurjano/boha.html

    The function is defined by:
        :math:`f(x, y) = x^2 + 2y^2 -0.3cos(3\pi x)-0.4cos(4\pi y)+0.7`

    It has one local minimum at:
        :math:`(x, y) = (0, 0)` where `f(x, y) = 0`.
    """
    def __init__(self):
        pass

    def get_tuner_params(self):
        return {'maximize': False}

    @staticmethod
    def get_tunable():
        x = IntHyperParam(min=-100, max=100)
        y = IntHyperParam(min=-100, max=100)

        return Tunable({'x': x, 'y': y})

    def score(self, x, y):
        z = np.cos(3 * np.pi * x)
        return -1 * (x**2 + 2 * y**2 - 0.3 * z - 0.4 * np.cos(4 * np.pi * y) + 0.7)
