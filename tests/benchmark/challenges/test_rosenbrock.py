# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import call, patch

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

    @patch('btb.benchmark.challenges.rosenbrock.IntHyperParam')
    @patch('btb.benchmark.challenges.rosenbrock.Tunable')
    def test_get_tunable(self, mock_tunable, mock_inthyperparam):
        # setup
        mock_inthyperparam.side_effect = [1, 2]
        mock_tunable.return_value = 'tunable'

        # run
        result = Rosenbrock().get_tunable()

        # assert
        assert result == 'tunable'
        assert mock_inthyperparam.call_args_list == [call(min=-50, max=50), call(min=-50, max=50)]
        mock_tunable.assert_called_once_with({'x': 1, 'y': 2})

    @patch('btb.benchmark.challenges.rosenbrock.IntHyperParam')
    @patch('btb.benchmark.challenges.rosenbrock.Tunable', autosepc=True)
    def test_get_tunable_custom_min_max(self, mock_tunable, mock_inthyperparam):
        # setup
        mock_inthyperparam.side_effect = [1, 2]
        instance = Rosenbrock(min_x=1, max_x=2, min_y=3, max_y=4)

        # run
        result = instance.get_tunable()

        # assert
        assert result == mock_tunable.return_value
        mock_inthyperparam.assert_has_calls([call(min=1, max=2), call(min=3, max=4)])
        mock_tunable.assert_called_once_with({'x': 1, 'y': 2})

    def test_evaluate(self):
        # run
        result = Rosenbrock().evaluate(1, 2)
        result_2 = Rosenbrock().evaluate(1, 1)

        # assert
        assert result == -100
        assert result_2 == 0
