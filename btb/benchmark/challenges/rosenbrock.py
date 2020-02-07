# -*- coding: utf-8 -*-

from btb.benchmark.challenges.challenge import Challenge


class Rosenbrock(Challenge):
    """Rosenbrock Challenge.

    This challenge represents the Rosenbrock function, this is a non-convex function, introduced by
    Howard H. Rosenbrock in 1960, which is used as a performance test problem for optimization
    algorithms.[1] It is also known as Rosenbrock's valley or Rosenbrock's banana function.

    The global minimum is inside a long, narrow, parabolic shaped flat valley. To find the valley
    is trivial. To converge to the global minimum, however, is difficult.

    Reference:
        https://en.wikipedia.org/wiki/Rosenbrock_function

    The function is defined by:
        :math:`f(x, y) = (a - x)^2 + b(y - x ^2)^2`

    It has a global minimum at:
        :math:`(x, y) = (a, a^2)` where `f(x, y) = 0`. Usually these parameters are set such that
        :math:`a = 1` and :math:`b = 100`. Only in the trivial case where :math:`a = 0` is the
        function symmetric and the minimum at the origin.

    Args:
        a (int):
            Number that ``a`` will take in the function. Defaults to 1.
        b (int):
            Number that ``b`` will take in the function. Defaults to 100.
        min_x (int):
            Minimum number that the hyperparameter can propouse for ``x``. Defaults to -50.
        max_x (int):
            Maximum number that the hyperparameter can propouse for ``x``. Defaults to 50.
        min_y (int):
            Minimum number that the hyperparameter can propouse for ``y``. Defaults to -50.
        max_y (int):
            Maximum number that the hyperparameter can propouse for ``y``. Defaults to 50.
    """
    def __init__(self, a=1, b=100, min_x=-50, max_x=50, min_y=-50, max_y=50):
        self.a = a
        self.b = b
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def get_tunable_hyperparameters(self):
        return {
            'x': {
                'type': 'int',
                'range': [self.min_x, self.max_x],
                'default': None
            },
            'y': {
                'type': 'int',
                'range': [self.min_y, self.max_y],
                'default': None
            }
        }

    def evaluate(self, x, y):
        return -1 * ((self.a - x)**2 + self.b * (y - x**2)**2)
