# -*- coding: utf-8 -*-

from unittest import TestCase

from btb.benchmark.challenges.bohachevsky import Bohachevsky


class TestBohachevsky(TestCase):

    def test___init__default(self):
        # run
        bohachevsky = Bohachevsky()

        # assert
        assert bohachevsky.min_x == -100
        assert bohachevsky.max_x == 100
        assert bohachevsky.min_y == -100
        assert bohachevsky.max_y == 100

    def test___init__custom(self):
        # run
        bohachevsky = Bohachevsky(min_x=4, max_x=5, min_y=6, max_y=7)

        # assert
        assert bohachevsky.min_x == 4
        assert bohachevsky.max_x == 5
        assert bohachevsky.min_y == 6
        assert bohachevsky.max_y == 7

    def test_get_tunable_hyperparameters(self):
        # run
        result = Bohachevsky().get_tunable_hyperparameters()

        # assert
        expected_result = {
            'x': {
                'type': 'int',
                'range': [-100, 100],
                'default': None
            },
            'y': {
                'type': 'int',
                'range': [-100, 100],
                'default': None
            }
        }

        assert result == expected_result

    def test_get_tunable_hyperparameters_custom_min_max(self):
        # run
        result = Bohachevsky(min_x=1, max_x=2, min_y=3, max_y=4).get_tunable_hyperparameters()

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
        result = Bohachevsky().evaluate(1, 2)
        result_2 = Bohachevsky().evaluate(0, 0)

        # assert
        assert result == -9.6
        assert result_2 == 0
