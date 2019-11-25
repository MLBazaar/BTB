from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from btb.benchmark import benchmark


class TestBenchmark(TestCase):

    def test_benchmark_challenges_callable(self):
        # setup
        function = MagicMock(return_value=0.1, __name__='test_function')
        challenge = MagicMock()
        challenge.return_value.get_tunable.return_value = 'tunable'

        # run
        result = benchmark(function, challenges=challenge)

        # assert
        expected_result = pd.DataFrame({
            'MagicMock': [0.1],
            'Mean': [0.1],
            'Std': [0.0],
        })

        expected_result.index = ['test_function']

        function.assert_called_once_with(challenge.return_value.evaluate, 'tunable', 1000)
        challenge.return_value.get_tunable.assert_called_once_with()

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1),
        )

    def test_benchmark_challenges_tuple(self):
        # setup
        function_a = MagicMock(return_value=0.1, __name__='function_a')
        function_b = MagicMock(return_value=0.1, __name__='function_b')
        candidates = (function_a, function_b)
        challenge = MagicMock()
        challenge.return_value.get_tunable.return_value = 'tunable'

        # run
        result = benchmark(candidates, challenges=challenge)

        # assert
        expected_result = pd.DataFrame({
            'MagicMock': [0.1, 0.1],
            'Mean': [0.1, 0.1],
            'Std': [0.0, 0.0],
        })

        expected_result.index = ['function_a', 'function_b']

        function_a.assert_called_once_with(challenge.return_value.evaluate, 'tunable', 1000)
        function_b.assert_called_once_with(challenge.return_value.evaluate, 'tunable', 1000)
        challenge.return_value.get_tunable.assert_called_once_with()

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1),
        )

    def test_benchmark_challenges_not_list(self):

        # setup
        function = MagicMock(return_value=0.1)
        tuner_function = {'test': function}
        challenge = MagicMock()
        challenge.return_value.get_tunable.return_value = 'tunable'

        # run
        result = benchmark(tuner_function, challenges=challenge)

        # assert
        expected_result = pd.DataFrame({
            'MagicMock': [0.1],
            'Mean': [0.1],
            'Std': [0.0],
        })

        expected_result.index = ['test']

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
            'MagicMock': [0.1],
            'Mean': [0.1],
            'Std': [0.0],
        })

        expected_result.index = ['test']

        function.assert_called_once_with(challenge.return_value.evaluate, 'tunable', 1000)
        challenge.return_value.get_tunable.assert_called_once_with()

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1),
        )

    def test_benchmark_candidates_not_dict_not_callable(self):
        # setup
        with self.assertRaises(TypeError):
            benchmark(1)
