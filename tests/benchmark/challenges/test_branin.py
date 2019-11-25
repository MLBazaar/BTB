# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import call, patch

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

    @patch('btb.benchmark.challenges.branin.FloatHyperParam')
    @patch('btb.benchmark.challenges.branin.Tunable')
    def test_get_tunable(self, mock_tunable, mock_floathyperparam):
        # setup
        mock_floathyperparam.side_effect = [1, 2]
        mock_tunable.return_value = 'tunable'

        # run
        result = Branin().get_tunable()

        # assert
        expected_calls = [call(min=-5.0, max=10.0), call(min=0.0, max=15.0)]
        assert result == 'tunable'
        assert mock_floathyperparam.call_args_list == expected_calls
        mock_tunable.assert_called_once_with({'x': 1, 'y': 2})

    @patch('btb.benchmark.challenges.branin.FloatHyperParam')
    @patch('btb.benchmark.challenges.branin.Tunable', autospec=True)
    def test_get_tunable_custom_min_max(self, mock_tunable, mock_floathyperparam):
        # setup
        mock_floathyperparam.side_effect = [1, 2]
        instance = Branin(min_x=1, max_x=2, min_y=3, max_y=4)

        # run
        result = instance.get_tunable()

        # assert
        assert result == mock_tunable.return_value
        mock_floathyperparam.assert_has_calls([call(min=1, max=2), call(min=3, max=4)])
        mock_tunable.assert_called_once_with({'x': 1, 'y': 2})

    def test_evaluate(self):
        # run
        result = Branin().evaluate(-np.pi, 12.275)
        result_2 = Branin().evaluate(np.pi, 2.275)
        result_3 = Branin().evaluate(10, 1)

        # assert
        assert result.round(6) == -0.397887
        assert result_2.round(6) == -0.397887
        assert result_3.round(6) == -5.954976
