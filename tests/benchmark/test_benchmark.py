from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from btb.benchmark import benchmark


class TestBenchmark(TestCase):

    def test_benchmark_challenges_not_list(self):

        # setup
        tuner_function = MagicMock(return_value='score')
        challenge = MagicMock(__name__='challenge')
        challenge.return_value.get_tunable.return_value = 'tunable'

        # assert
        result = benchmark(tuner_function, challenges=challenge)

        # run
        expected_result = pd.DataFrame({
            'score': ['score'],
            'iterations': [1000],
            'challenge': 'challenge'
        })

        tuner_function.assert_called_once_with(challenge.return_value.score, 'tunable', 1000)
        challenge.return_value.get_tunable.assert_called_once_with()
        pd.testing.assert_frame_equal(result, expected_result)

    def test_benchmark_challenges_list(self):

        # setup
        tuner_function = MagicMock(return_value='score')
        challenge = MagicMock(__name__='challenge')
        challenge.return_value.get_tunable.return_value = 'tunable'

        # assert
        result = benchmark(tuner_function, challenges=[challenge])

        # run
        expected_result = pd.DataFrame({
            'score': ['score'],
            'iterations': [1000],
            'challenge': 'challenge'
        })

        tuner_function.assert_called_once_with(challenge.return_value.score, 'tunable', 1000)
        challenge.return_value.get_tunable.assert_called_once_with()
        pd.testing.assert_frame_equal(result, expected_result)
