from btb.selection import Selector
import random
import numpy as np


class UCB1(Selector):
    """
    The most common Selector implementation.
    Uses Upper Confidence Bound 1 algorithm (UCB1) for bandit selection.
    """
    def bandit(self, choice_rewards):
        """
        MAB method which chooses the arm for which the upper confidence bound
        (UCB) of expected reward is greatest.
        An explanation is here:
        https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf
        """
        # count the total number of times all "levers" have been "pulled" so far.
        # don't let the value go below 1, so that log() and division still work.
        total_pulls = max(sum(len(r) for r in choice_rewards.values()), 1)
        scores = {}

        # shuffle the arms so that if all else is equal, we don't choose the same
        # one every time
        choices = choice_rewards.items()
        random.shuffle(choices)

        for choice, rewards in choices:
            # count the number of pulls for this choice, with a floor of 1
            choice_pulls = max(len(rewards), 1)

            # compute the 2-stdev error for the estimate of this choice
            error = np.sqrt(2.0 * np.log(total_pulls) / choice_pulls)

            # compute the average reward, or default to 0
            avg_reward = np.mean(rewards) if rewards else 0

            # this choice's score is the upper bound of what we think is possible
            scores[choice] = avg_reward + error

        best_choice = sorted(scores.keys(), key=scores.get)[-1]
        return best_choice
