from unittest import TestCase

from btb.selection.best import BestKReward


class TestBestKReward(TestCase):

    # METHOD: __init__(self, choices, **kwargs)
    # VALIDATE:
    #     * attribute values
    # TODO:
    #     * kwargs can be safely removed from the method signature
    #     * 'k' should be made an explicit parameter

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
        assert rewards == [0.83, 0.0, 0.86, 0.9]

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * Returned values for multiple cases
    #     * arguments passed to bandit
    # TODO:
    #     * This code can be simplified to be more easy to understand

    # Work in progress
    #def test_select(self):

    #    # Set-up
    #    selector = Selector(['RF', 'SVM'])

    #    # Run
    #    choice_scores = {
    #        'DT': [0.7, 0.73, 0.75],
    #        'RF': [0.8, 0.83, 0.85],
    #        'SVM': [0.9, 0.93, 0.95]
    #    }
    #    best = selector.select(choice_scores)

    #    # Assert
    #    assert best == 'SVM'


class TestBestKVelocity(TestCase):

    # METHOD: compute_rewards(self, scores)
    # VALIDATE:
    #     * returned values
    pass
