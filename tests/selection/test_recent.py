from unittest import TestCase
from unittest.mock import patch

import numpy as np

from btb.selection.recent import RecentKReward, RecentKVelocity


class TestRecentKReward(TestCase):

    # METHOD: __init__(self, choices, **kwargs)
    # VALIDATE:
    #     * attribute values

    def test___init__(self):

        # Run
        selector = RecentKReward(['RF', 'SVM'], k=3)

        # Assert
        assert selector.choices == ['RF', 'SVM']
        assert selector.k == 3

    # METHOD: compute_rewards(self, scores)
    # VALIDATE:
    #     * returned values

    def test_compute_rewards(self):
        """Last scores are zeroed"""

        # Set-up
        selector = RecentKReward(['RF', 'SVM'], k=3)

        # Run
        scores = [0.5, 0.6, 0.75, 0.8, 0.9]
        rewards = selector.compute_rewards(scores)

        # Assert
        assert rewards == [0.5, 0.6, 0.75, 0., 0.]

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * Returned values for multiple cases
    #     * arguments passed to bandit
    # TODO:
    #     * This code can be simplified to be more easy to understand

    @patch('btb.selection.recent.RecentKReward.bandit')
    def test_select_more_scores_than_k_min(self, bandit_mock):
        """If min length is gt k_min, self.compute_rewards is used.

        In this case, we expect the lower scores to be zeroed.
        """

        # Set-up
        selector = RecentKReward(['RF', 'SVM'])

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

        choice_rewards = {
            'RF': [0.8, 0.85, 0.],
            'SVM': [0.9, 0.95, 0.],
        }
        bandit_mock.assert_called_once_with(choice_rewards)

    @patch('btb.selection.recent.RecentKReward.bandit')
    def test_select_less_scores_than_k_min(self, bandit_mock):
        """If min length is lt k_min, super().compute_rewards is used.

        In this case, we expect the socres to be returned as they are.
        """

        # Set-up
        selector = RecentKReward(['RF', 'SVM'])

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


class TestRecentKVelocity(TestCase):

    # METHOD: compute_rewards(self, scores)
    # VALIDATE:
    #     * returned values

    def test_compute_rewards_kt_k(self):
        """Less scores than self.k: No padding"""

        # Set-up
        selector = RecentKVelocity(['RF', 'SVM'], k=5)

        # Run
        scores = [0.5, 0.6, 0.75, 0.8]
        rewards = selector.compute_rewards(scores)

        # Assert
        np.testing.assert_allclose(rewards, [0.05, 0.15, 0.1])

    def test_compute_rewards_gt_k(self):
        """More scores than self.k: padding"""

        # Set-up
        selector = RecentKVelocity(['RF', 'SVM'], k=3)

        # Run
        scores = [0.1, 0.5, 0.6, 0.75, 0.8]
        rewards = selector.compute_rewards(scores)

        # Assert
        np.testing.assert_allclose(rewards, [0.05, 0.15, 0.1, 0., 0.])
