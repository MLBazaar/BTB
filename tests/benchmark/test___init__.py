# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from btb.benchmark import benchmark, evaluate_candidate


class TestBenchmark(TestCase):

    def test_evaluate_candidate_challenge_is_instance(self):
        # setup
        candidate = MagicMock()
        challenge = MagicMock()
        challenge.__str__.return_value = 'test_challenge'

        # run
        result = evaluate_candidate('test_candidate', candidate, challenge, 10)

        # assert
        challenge.get_tunable_hyperparameters.assert_called_once_with()
        candidate.assert_called_once_with(
            challenge.evaluate,
            challenge.get_tunable_hyperparameters.return_value,
            10
        )

        expected_result = [
            {
                'challenge': 'test_challenge',
                'candidate': 'test_candidate',
                'score': candidate.return_value
            }
        ]

        assert result == expected_result

    @patch('btb.benchmark.evaluate_candidate')
    def test_benchmark_challenges_callable(self, mock_evaluate_candidate):
        # setup
        mock_evaluate_candidate.return_value = [{
            'challenge': 'test_challenge',
            'candidate': 'test_candidate',
            'score': 1.0
        }]

        candidate = MagicMock(__name__='test_candidate')
        challenge = MagicMock()

        # run
        result = benchmark(candidate, challenges=challenge)

        # assert
        expected_result = pd.DataFrame({
            'test_challenge': [1.0],
            'Mean': [1.0],
            'Std': [0.0],
        })

        expected_result.index = ['test_candidate']

        mock_evaluate_candidate.assert_called_once_with(
            'test_candidate',
            candidate,
            [challenge],
            1000
        )

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1),
        )

    @patch('btb.benchmark.evaluate_candidate')
    def test_benchmark_challenges_tuple(self, mock_evaluate_candidate):
        # setup
        mock_evaluate_candidate.side_effect = [
            [{'challenge': 'test_challenge', 'candidate': 'candidate_a', 'score': 1.0}],
            [{'challenge': 'test_challenge', 'candidate': 'candidate_b', 'score': 1.0}],
        ]

        candidate_a = MagicMock(return_value=0.1, __name__='candidate_a')
        candidate_b = MagicMock(return_value=0.1, __name__='candidate_b')
        candidates = (candidate_a, candidate_b)
        challenge = MagicMock()

        # run
        result = benchmark(candidates, challenges=challenge)

        # assert
        expected_result = pd.DataFrame({
            'test_challenge': [1.0, 1.0],
            'Mean': [1.0, 1.0],
            'Std': [0.0, 0.0],
        })

        expected_result.index = ['candidate_a', 'candidate_b']

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            expected_result.sort_index(axis=1),
        )

    def test_benchmark_candidates_not_dict_not_callable(self):
        # setup
        with self.assertRaises(TypeError):
            benchmark(1)
