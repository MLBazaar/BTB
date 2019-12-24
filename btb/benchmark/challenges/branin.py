# -*- coding: utf-8 -*-

import numpy as np

from btb.benchmark.challenges.challenge import Challenge

B = 5.1 / (4 * pow(np.pi, 2))
C = 5 / np.pi
T = 1 / (8 * np.pi)


class Branin(Challenge):
    r"""Branin challenge.

    The Branin, or Branin-Hoo, function is commonly used as a test function for metamodeling
    in computer experiments, especially in the context of optimization. This function takes as
    input ``x`` and ``y`` and has six constants that are named as ``a, b, c, r, s, t``.

    Reference:
        https://uqworld.org/t/branin-function/53https://uqworld.org/t/branin-function/53

    The function is defined by:
       :math:`f(x, y)=\left(y-\frac{5.1 x^2_1}{4 \pi^2}+\frac{5 x}{\pi}-6\right)^2+10\left(1-
       \frac{1}{8 \pi}\right) \cos(x)+10`

    It has a global minimum, with the default `a, b, c, r, s, t` at the following three points:
        :math:`f(x, y) = 0.397887` at :math:`x, y = (-\pi, 12.275), (\pi, 2.275)` and
        :math:`(9.42478, 2.475)`

    Args:
        a (int):
            Constant value for ``a``. Defaults to 1.
        b (float or int):
            Constant value for ``b``. Defaults to :math:`5.1 / (4*\pi^2)`.
        c (float or int):
            Constant value for ``c``. Defaults to :math:`5 / \pi`.
        r (int):
            Constant value for ``r``. Defaults to 6
        s (int):
            Constant value for ``s``. Defaults to 10
        t (float or int):
            Constant value for ``t``. Defaults to :math:`1 / 8*\pi`.
    """
    def __init__(self, a=1, b=B, c=C, r=6, s=10, t=T,
                 min_x=-5.0, max_x=10.0, min_y=0.0, max_y=15.0):
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def get_tunable_hyperparameters(self):
        return {
            'x': {
                'type': 'float',
                'range': [self.min_x, self.max_x],
                'default': None
            },
            'y': {
                'type': 'float',
                'range': [self.min_y, self.max_y],
                'default': None
            }
        }

    def evaluate(self, x, y):
        z = (y - self.b * x**2 + self.c * x - self.r)**2
        return -1 * (self.a * z + self.s * (1 - self.t) * np.cos(x) + self.s)
