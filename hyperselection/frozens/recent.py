from hyperselection.frozens import FrozenSelector, UCB1
from hyperselection.bandit import ucb1_bandit
import random
import numpy as np

# minimum number of examples required for ALL frozen
# sets to have evaluated in order to use recent K optimizations
K_MIN = 2


class RecentKReward(FrozenSelector):
    def __init__(self, choices, **kwargs):
        """
        Needs:
            k: number of best scores to consider
        """
        super(RecentKReward, self).__init__(choices, **kwargs)
        self.k = kwargs.pop('k', K_MIN)
        self.ucb1 = UCB1(choices, **kwargs)

    def select(self, choice_scores):
        """
        Keeps the frozen set counts intact but only uses the top k learner's
        scores for usage in rewards for the bandit calculation
        """
        # if we don't have enough scores to do K-selection, fall back to UCB1
        if min([len(s) for s in choice_scores.values()]) < K_MIN:
            return self.ucb1.select(choice_scores)

        recent_k_scores = {}
        # all scores are already in chronological order
        for choice, scores in choice_scores.items():
            if choice not in self.choices:
                continue
            zeros = (len(scores) - self.k) * [0]
            recent_k_scores[choice] = scores[-self.k:] + zeros

        return ucb1_bandit(recent_k_scores)


class RecentKVelocity(FrozenSelector):
    def __init__(self, **kwargs):
        """
        Needs:
            k: number of best scores to consider
        """
        super(RecentKVelocity, self).__init__(choices, **kwargs)
        self.k = kwargs.get('k', K_MIN)
        self.ucb1 = UCB1(choices, **kwargs)

    def select(self, choice_scores):
        """
        Keeps the frozen set counts intact but only uses the top k learner's
        velocities over their last for usage in rewards for the bandit
        calculation
        """
        # if we don't have enough scores to do K-selection, fall back to UCB1
        if min([len(s) for s in choice_scores.values()]) < K_MIN:
            return self.ucb1.select(choice_scores)

        recent_k_velocities = {}
        # all scores are already in chronological order
        for choice, scores in choice_scores.items():
            if choice not in self.choices:
                continue
            # take the k + 1 most recent scores so we can get k velocities
            recent_scores = scores[:-self.k-2:-1]
            velocities = [recent_scores[i] - recent_scores[i+1] for i in
                          range(len(recent_scores) - 1)])
            # pad the list out with zeros, so the length of the list is
            # maintained
            zeros = (len(s) - self.k) * [0]
            recent_k_velocities[c] = velocities + zeros

        return ucb1_bandit(recent_k_velocities)
