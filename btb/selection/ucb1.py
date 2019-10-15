import random

import numpy as np

from btb.selection.selector import Selector


class UCB1(Selector):
    """UCB1 selector

    Uses Upper Confidence Bound 1 algorithm (UCB1) for bandit selection.

    See also::

       Auer, Peter et al. "Finite-time Analysis of the Multiarmed Bandit Problem."
       Machine Learning 47 (2002): 235-256.
    """

    def _shuffle(self, iterable):
        iterable = list(iterable)
        inds = list(range(len(iterable)))
        random.shuffle(inds)
        for i in inds:
            yield iterable[i]

    def bandit(self, choice_rewards):
        """
        Multi-armed bandit method which chooses the arm for which the upper
        confidence bound (UCB) of expected reward is greatest.

        If there are multiple arms with the same UCB1 index, then one is chosen
        at random.

        An explanation is here:
        https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf
        """

        # count the larger of 1 and the total number of arm pulls
        total_pulls = max(1, sum(len(r) for r in choice_rewards.values()))

        def ucb1(choice):
            rewards = choice_rewards[choice]
            choice_pulls = max(len(rewards), 1)
            average_reward = np.nanmean(rewards) if len(rewards) else 0
            error = np.sqrt(2.0 * np.log(total_pulls) / choice_pulls)
            return average_reward + error

        return max(self._shuffle(choice_rewards), key=ucb1)
