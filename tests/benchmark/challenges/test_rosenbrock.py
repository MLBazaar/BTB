# -*- coding: utf-8 -*-

from unittest import TestCase

from btb.benchmark.challenges.rosenbrock import Rosenbrock


class TestRosenbrock(TestCase):

    def test___init__default(self):
        # run
        rosenbrock = Rosenbrock()

        # assert
        assert rosenbrock.a == 1
        assert rosenbrock.b == 100
        assert rosenbrock.min_x == -50
        assert rosenbrock.max_x == 50
        assert rosenbrock.min_y == -50
        assert rosenbrock.max_y == 50

    def test___init__custom(self):
        # run
        rosenbrock = Rosenbrock(a=2, b=3, min_x=4, max_x=5, min_y=6, max_y=7)

        # assert
        assert rosenbrock.a == 2
        assert rosenbrock.b == 3
        assert rosenbrock.min_x == 4
        assert rosenbrock.max_x == 5
        assert rosenbrock.min_y == 6
        assert rosenbrock.max_y == 7

    def test_get_tunable_hyperparameters(self):
        # run
        result = Rosenbrock().get_tunable_hyperparameters()

        # assert
        expected_result = {
            'x': {
                'type': 'int',
                'range': [-50, 50],
                'default': None
            },
            'y': {
                'type': 'int',
                'range': [-50, 50],
                'default': None
            }
        }

        assert result == expected_result

    def test_get_tunable_hyperparameters_custom_min_max(self):
        # run
        result = Rosenbrock(min_x=1, max_x=2, min_y=3, max_y=4).get_tunable_hyperparameters()

        # assert
        expected_result = {
            'x': {
                'type': 'int',
                'range': [1, 2],
                'default': None
            },
            'y': {
                'type': 'int',
                'range': [3, 4],
                'default': None
            }
        }

        assert result == expected_result

    def test_evaluate(self):
        # run
        result = Rosenbrock().evaluate(1, 2)
        result_2 = Rosenbrock().evaluate(1, 1)

        # assert
        assert result == -100
        assert result_2 == 0
