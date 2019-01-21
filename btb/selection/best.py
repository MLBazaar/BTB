import logging

import numpy as np

from btb.selection.ucb1 import UCB1

# the minimum number of scores that each choice must have in order to use best-K
# optimizations. If not all choices meet this threshold, default UCB1 selection
# will be used.
K_MIN = 2

logger = logging.getLogger('btb')


class BestKReward(UCB1):
    """Best K reward selector

    Computes the average reward from the past scores by using only the highest k scores. In
    implementation, the other scores are replaced with ``nan``s such that they still factor into
    the number of arm pulls.

    Args:
        k (int): number of best scores to consider
    """

    def __init__(self, choices, k=K_MIN):
        super(BestKReward, self).__init__(choices)
        self.k = k

    def compute_rewards(self, scores):
        """Retain the K best scores, and replace the rest with nans"""
        if len(scores) > self.k:
            scores = np.copy(scores)
            inds = np.argsort(scores)[:-self.k]
            scores[inds] = np.nan

        return list(scores)

    def select(self, choice_scores):
        """Select a choice using the K best scores

        Keeps the choice counts intact, but only let the bandit see the top k learners' scores.
        If there is not enough score history to do K-selection, use the default UCB1 reward
        function.
        """
        min_num_scores = min(len(s) for s in choice_scores.values())
        if min_num_scores >= K_MIN:
            logger.info(
                '{klass}: using Best K bandit selection'
                .format(klass=type(self).__name__))
            compute_rewards = self.compute_rewards
        else:
            logger.warning(
                '{klass}: Not enough choices to do K-selection; using plain UCB1'
                .format(klass=type(self).__name__))
            compute_rewards = super(BestKReward, self).compute_rewards

        # convert the raw scores list for each choice to a "rewards" list
        choice_rewards = {
            choice: compute_rewards(choice_scores[choice])
            for choice in choice_scores
            if choice in self.choices
        }

        return self.bandit(choice_rewards)


class BestKVelocity(BestKReward):
    """Best K velocity selector"""

    def compute_rewards(self, scores):
        """Compute the velocity of the best scores

        The velocities are the k distances between the k+1 best scores.
        """
        k = self.k
        m = max(len(scores) - k, 0)
        best_scores = sorted(scores)[-k - 1:]
        velocities = np.diff(best_scores)
        nans = np.full(m, np.nan)
        return list(velocities) + list(nans)
