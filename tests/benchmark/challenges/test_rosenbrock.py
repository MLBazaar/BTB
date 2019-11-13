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
        assert rosenbrock.b == 1

    def test___init__custom(self):
        # run
        rosenbrock = Rosenbrock(a=2, b=3)

        # assert
        assert rosenbrock.a == 2
        assert rosenbrock.b == 3

    @patch('btb.benchmark.challenges.rosenbrock.IntHyperParam')
    @patch('btb.benchmark.challenges.rosenbrock.Tunable')
    def test_get_tunable(self, mock_tunable, mock_inthyperparam):
        # setup
        mock_inthyperparam.side_effect = [1, 2]
        mock_tunable.return_value = 'tunable'

        # run
        result = Rosenbrock.get_tunable()

        # assert
        assert result == 'tunable'
        mock_inthyperparam.call_args_list == [call(min=-50, max=50), call(min=-50, max=50)]
        mock_tunable.assert_called_once_with({'x': 1, 'y': 2})

    def test_score(self):
        # run
        result = Rosenbrock().score(1, 2)
        result_2 = Rosenbrock().score(1, 1)

        # assert
        assert result == -1
        assert result_2 == 0
