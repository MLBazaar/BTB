# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from btb.benchmark.challenges.branin import Branin


class TestBranin(TestCase):

    def test___init__default(self):
        # run
        branin = Branin()

        # assert
        assert branin.a == 1
        assert branin.b == 5.1 / (4 * pow(np.pi, 2))
        assert branin.c == 5 / np.pi
        assert branin.r == 6
        assert branin.s == 10
        assert branin.t == 1 / (8 * np.pi)
        assert branin.min_x == -5.0
        assert branin.max_x == 10.0
        assert branin.min_y == 0.0
        assert branin.max_y == 15.0

    def test___init__custom(self):
        # run
        branin = Branin(a=1, b=2, c=3, r=4, s=5, t=6, min_x=4, max_x=5, min_y=6, max_y=7)

        # assert
        assert branin.a == 1
        assert branin.b == 2
        assert branin.c == 3
        assert branin.r == 4
        assert branin.s == 5
        assert branin.t == 6
        assert branin.min_x == 4
        assert branin.max_x == 5
        assert branin.min_y == 6
        assert branin.max_y == 7

    def test_get_tunable_hyperparameters(self):
        # run
        result = Branin().get_tunable_hyperparameters()

        # assert
        expected_result = {
            'x': {
                'type': 'float',
                'range': [-5.0, 10.0],
                'default': None
            },
            'y': {
                'type': 'float',
                'range': [0.0, 15.0],
                'default': None
            }
        }

        result == expected_result

    def test_get_tunable_custom_min_max(self):
        # run
        result = Branin(min_x=1, max_x=2, min_y=3, max_y=4).get_tunable_hyperparameters()

        # assert
        expected_result = {
            'x': {
                'type': 'float',
                'range': [1, 2],
                'default': None
            },
            'y': {
                'type': 'float',
                'range': [3, 4],
                'default': None
            }
        }

        result == expected_result

    def test_evaluate(self):
        # run
        result = Branin().evaluate(-np.pi, 12.275)
        result_2 = Branin().evaluate(np.pi, 2.275)
        result_3 = Branin().evaluate(10, 1)

        # assert
        assert result.round(6) == -0.397887
        assert result_2.round(6) == -0.397887
        assert result_3.round(6) == -5.954976
