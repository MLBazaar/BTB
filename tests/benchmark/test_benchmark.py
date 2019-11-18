from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from btb.benchmark import benchmark


class TestBenchmark(TestCase):

    def test_benchmark_challenges_not_list(self):

        # setup
        function = MagicMock(return_value=0.1)
        tuner_function = {'test': function}
        challenge = MagicMock(__name__='challenge')
        challenge.return_value.get_tunable.return_value = 'tunable'

        # run
        result = benchmark(tuner_function, challenges=challenge)

        # assert
        expected_result = pd.DataFrame({
            'avg': [0.1],
            'challenge': 'challenge',
            'iterations': [1000],
            'tuner': 'test',
            'score': [0.1],
        })

        function.assert_called_once_with(challenge.return_value.evaluate, 'tunable', 1000)
        challenge.return_value.get_tunable.assert_called_once_with()

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1),
        )

    def test_benchmark_challenges_list(self):

        # setup
        function = MagicMock(return_value=0.1)
        tuner_function = {'test': function}
        challenge = MagicMock(__name__='challenge')
        challenge.return_value.get_tunable.return_value = 'tunable'

        # assert
        result = benchmark(tuner_function, challenges=[challenge])

        # run
        expected_result = pd.DataFrame({
            'challenge': 'challenge',
            'tuner': 'test',
            'score': [0.1],
            'iterations': [1000],
            'avg': [0.1],
        })

        function.assert_called_once_with(challenge.return_value.evaluate, 'tunable', 1000)
        challenge.return_value.get_tunable.assert_called_once_with()

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1),
        )
