from unittest import TestCase
from unittest.mock import patch

import numpy as np

from btb.selection.best import BestKReward, BestKVelocity


class TestBestKReward(TestCase):

    # METHOD: __init__(self, choices, **kwargs)
    # VALIDATE:
    #     * attribute values

    def test___init__(self):

        # Run
        selector = BestKReward(['RF', 'SVM'], k=3)

        # Assert
        assert selector.choices == ['RF', 'SVM']
        assert selector.k == 3

    # METHOD: compute_rewards(self, scores)
    # VALIDATE:
    #     * returned values
    #     * check the case whete len(scores) < self.k

    def test_compute_rewards_lt_k(self):

        # Set-up
        selector = BestKReward(['RF', 'SVM'], k=3)

        # Run
        scores = [0.8, 0.9]
        rewards = selector.compute_rewards(scores)

        # Assert
        assert rewards == scores

    def test_compute_rewards_gt_k(self):

        # Set-up
        selector = BestKReward(['RF', 'SVM'], k=3)

        # Run
        scores = [0.83, 0.8, 0.86, 0.9]
        rewards = selector.compute_rewards(scores)

        # Assert
        np.testing.assert_array_equal(rewards, [0.83, np.nan, 0.86, 0.9])

    def test_compute_rewards_duplicates(self):
        k = 3
        choices = ['RF', 'SVM']
        scores = [0.7, 0.8, 0.7, 0.1, 0.8, 0.7]
        selector = BestKReward(choices, k=k)
        rewards = selector.compute_rewards(scores)
        np.testing.assert_array_equal(
            np.sort(rewards),
            np.sort([np.nan, np.nan, np.nan, 0.7, 0.8, 0.8])
        )

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * Returned values for multiple cases
    #     * arguments passed to bandit
    # TODO:
    #     * This code can be simplified to be more easy to understand
    # NOTES:
    #     * Should this be >= self.k instead of >= K_MIN?

    @patch('btb.selection.best.BestKReward.bandit')
    def test_select_more_scores_than_k_min(self, bandit_mock):
        """If min length is gt k_min, self.compute_rewards is used.

        In this case, we expect the lower scores to be nan.
        """

        # Set-up
        selector = BestKReward(['RF', 'SVM'])

        bandit_mock.return_value = 'SVM'

        # Run
        choice_scores = {
            'DT': [0.7, 0.75, 0.73],
            'RF': [0.8, 0.85, 0.83],
            'SVM': [0.9, 0.95, 0.93],
        }
        best = selector.select(choice_scores)

        # Assert
        assert best == 'SVM'

        choice_rewards_expected = {
            'RF': [np.nan, 0.85, 0.83],
            'SVM': [np.nan, 0.95, 0.93],
        }
        # TODO
        (choice_rewards, ), _ = bandit_mock.call_args
        assert choice_rewards.keys() == choice_rewards_expected.keys()
        for k in choice_rewards:
            assert k in choice_rewards_expected
            np.testing.assert_array_equal(
                choice_rewards[k],
                choice_rewards_expected[k]
            )

    @patch('btb.selection.best.BestKReward.bandit')
    def test_select_less_scores_than_k_min(self, bandit_mock):
        """If min length is lt k_min, super().compute_rewards is used.

        In this case, we expect the socres to be returned as they are.
        """

        # Set-up
        selector = BestKReward(['RF', 'SVM'])

        bandit_mock.return_value = 'SVM'

        # Run
        choice_scores = {
            'DT': [0.7, 0.75, 0.73],
            'RF': [0.8],
            'SVM': [0.9, 0.95, 0.93],
        }
        best = selector.select(choice_scores)

        # Assert
        assert best == 'SVM'

        choice_rewards = {
            'RF': [0.8],
            'SVM': [0.9, 0.95, 0.93],
        }
        bandit_mock.assert_called_once_with(choice_rewards)


class TestBestKVelocity(TestCase):

    # METHOD: compute_rewards(self, scores)
    # VALIDATE:
    #     * returned values

    def test_compute_rewards_kt_k(self):
        """Less scores than self.k: No padding"""

        # Set-up
        selector = BestKVelocity(['RF', 'SVM'], k=5)

        # Run
        scores = [0.5, 0.6, 0.75, 0.8]
        rewards = selector.compute_rewards(scores)

        # Assert
        np.testing.assert_allclose(
            np.sort(rewards),
            np.sort([0.05, 0.15, 0.1])
        )

    def test_compute_rewards_gt_k(self):
        """More scores than self.k: padding"""

        # Set-up
        selector = BestKVelocity(['RF', 'SVM'], k=3)

        # Run
        scores = [0.1, 0.5, 0.6, 0.75, 0.8]
        rewards = selector.compute_rewards(scores)

        # Assert
        np.testing.assert_allclose(
            np.sort(rewards),
            np.sort([0.05, 0.15, 0.1, np.nan, np.nan])
        )
