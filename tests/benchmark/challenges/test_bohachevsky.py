# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import call, patch

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

    @patch('btb.benchmark.challenges.bohachevsky.IntHyperParam')
    @patch('btb.benchmark.challenges.bohachevsky.Tunable')
    def test_get_tunable(self, mock_tunable, mock_inthyperparam):
        # setup
        mock_inthyperparam.side_effect = [1, 2]
        mock_tunable.return_value = 'tunable'

        # run
        result = Bohachevsky().get_tunable()

        # assert
        expected_calls = [call(min=-100, max=100), call(min=-100, max=100)]
        assert result == 'tunable'
        assert mock_inthyperparam.call_args_list == expected_calls
        mock_tunable.assert_called_once_with({'x': 1, 'y': 2})

    @patch('btb.benchmark.challenges.bohachevsky.IntHyperParam')
    @patch('btb.benchmark.challenges.bohachevsky.Tunable', autospec=True)
    def test_get_tunable_custom_min_max(self, mock_tunable, mock_inthyperparam):
        # setup
        mock_inthyperparam.side_effect = [1, 2]
        instance = Bohachevsky(min_x=1, max_x=2, min_y=3, max_y=4)

        # run
        result = instance.get_tunable()

        # assert
        assert result == mock_tunable.return_value
        mock_inthyperparam.assert_has_calls([call(min=1, max=2), call(min=3, max=4)])
        mock_tunable.assert_called_once_with({'x': 1, 'y': 2})

    def test_evaluate(self):
        # run
        result = Bohachevsky().evaluate(1, 2)
        result_2 = Bohachevsky().evaluate(0, 0)

        # assert
        assert result == -9.6
        assert result_2 == 0
