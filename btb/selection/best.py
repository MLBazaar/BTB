import logging
from builtins import range
from btb.selection import Selector, UCB1
import numpy as np

# the minimum number of scores that each choice must have in order to use best-K
# optimizations. If not all choices meet this threshold, default UCB1 selection
# will be used.
K_MIN = 2

logger = logging.getLogger('btb')


class BestKReward(UCB1):
    def __init__(self, choices, **kwargs):
        """
        Extra args:
            k: number of best scores to consider
        """
        super(BestKReward, self).__init__(choices, **kwargs)
        self.k = kwargs.pop('k', K_MIN)

    def compute_rewards(self, scores):
        """ Retain the K best scores, and replace the rest with zeros """
        if len(scores) > self.k:
            kth_best = sorted(scores, reverse=True)[self.k - 1]
            return [(s if s >= kth_best else 0.) for s in scores]
        else:
            return list(scores)

    def select(self, choice_scores):
        """
        Keeps the choice counts intact, but only let the bandit see the top k
        learners' scores.
        """
        # if we don't have enough scores to do K-selection, use the default UCB1
        # reward function
        min_num_scores = min([len(s) for s in choice_scores.values()])
        if min_num_scores >= K_MIN:
            logger.info('BestK: using Best K bandit selection')
            reward_func = self.compute_rewards
        else:
            logger.warn('BestK: Not enough choices to do K-selection; using plain UCB1')
            reward_func = super(BestKReward, self).compute_rewards

        # convert the raw scores list for each choice to a "rewards" list
        choice_rewards = {}
        for choice, scores in choice_scores.items():
            # only consider choices that this object was initialized with
            if choice not in self.choices:
                continue
            choice_rewards[choice] = reward_func(scores)

        return self.bandit(choice_rewards)


class BestKVelocity(BestKReward):
    def compute_rewards(self, scores):
        """
        Compute the "velocity" of (average distance between) the k+1 best
        scores. Return a list with those k velocities padded out with zeros so
        that the count remains the same.
        """
        # get the k + 1 best scores in descending order
        best_scores = sorted(scores, reverse=True)[:self.k+1]
        velocities = [best_scores[i] - best_scores[i+1]
                      for i in range(len(best_scores) - 1)]

        # pad the list out with zeros to maintain the lenghth of the list
        zeros = (len(scores) - self.k) * [0]
        return velocities + zeros
