# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from btb.benchmark import benchmark


class TestBenchmark(TestCase):

    @patch('btb.benchmark.evaluate_tuner')
    def test_benchmark_challenges_callable(self, mock_evaluate_tuner):
        # setup
        mock_evaluate_tuner.return_value = [{
            'challenge': 'test_challenge',
            'tuner': 'test_tuner',
            'score': 1.0
        }]

        tuner = MagicMock(__name__='test_tuner')
        challenge = MagicMock()

        # run
        result = benchmark(tuner, challenges=challenge)

        # assert
        expected_result = pd.DataFrame({
            'test_tuner': [1.0],
        })

        expected_result.index = ['test_challenge']

        mock_evaluate_tuner.assert_called_once_with(
            'test_tuner',
            tuner,
            [challenge],
            1000
        )

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1),
        )

    @patch('btb.benchmark.evaluate_tuner')
    def test_benchmark_challenges_tuple(self, mock_evaluate_tuner):
        # setup
        mock_evaluate_tuner.side_effect = [
            [{'challenge': 'test_challenge', 'tuner': 'tuner_a', 'score': 1.0}],
            [{'challenge': 'test_challenge', 'tuner': 'tuner_b', 'score': 1.0}],
        ]

        tuner_a = MagicMock(return_value=0.1, __name__='tuner_a')
        tuner_b = MagicMock(return_value=0.1, __name__='tuner_b')
        tuners = (tuner_a, tuner_b)
        challenge = MagicMock()

        # run
        result = benchmark(tuners, challenges=challenge)

        # assert
        expected_result = pd.DataFrame({
            'tuner_a': [1.0],
            'tuner_b': [1.0],
        })

        expected_result.index = ['test_challenge']

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1),
        )

    def test_benchmark_tuners_not_dict_not_callable(self):
        # setup
        with self.assertRaises(TypeError):
            benchmark(1)
