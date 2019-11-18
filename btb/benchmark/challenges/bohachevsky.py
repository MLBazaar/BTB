# -*- coding: utf-8 -*-

import numpy as np

from btb.benchmark.challenges.challenge import Challenge
from btb.tuning import Tunable
from btb.tuning.hyperparams import IntHyperParam


class Bohachevsky(Challenge):
    r"""Bohachevsky challenge.

    The Bohachevsky functions are bowl shape functions. This function is usually evaluated on the
    input domain :math:`x \epsilon [-100, 100], y \epsilon [-100, 100]`.

    Reference:
        https://www.sfu.ca/~ssurjano/boha.html

    The function is defined by:
        :math:`f(x, y) = x^2 + 2y^2 -0.3cos(3\pi x)-0.4cos(4\pi y)+0.7`

    It has one local minimum at:
        :math:`(x, y) = (0, 0)` where :math:`f(x, y) = 0`.

    Args:
        min_x (int):
            Minimum number that the hyperparameter can propouse for ``x``. Defaults to -100.
        max_x (int):
            Maximum number that the hyperparameter can propouse for ``x``. Defaults to 100.
        min_y (int):
            Minimum number that the hyperparameter can propouse for ``y``. Defaults to -100.
        max_y (int):
            Maximum number that the hyperparameter can propouse for ``y``. Defaults to 100.
    """
    def __init__(self, min_x=-100, max_x=100, min_y=-100, max_y=100):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def get_tunable(self):
        x = IntHyperParam(min=self.min_x, max=self.max_x)
        y = IntHyperParam(min=self.min_y, max=self.max_y)

        return Tunable({'x': x, 'y': y})

    def evaluate(self, x, y):
        z = 0.3 * np.cos(3 * np.pi * x)
        return -1 * (x**2 + 2 * y**2 - z - 0.4 * np.cos(4 * np.pi * y) + 0.7)
