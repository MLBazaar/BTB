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
        Multi-armed bandit method which chooses the arm for which the upper
        confidence bound (UCB) of expected reward is greatest.
        An explanation is here:
        https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf

        choice_rewards: a dict, {choice -> [rewards]}, mapping each potential
            choice to a series of "rewards" granted for its past behavior.

        Returns: the single choice (one of choice_rewards.keys()) which we
            expect will perform best.
        """
        # count the total number of times all "levers" have been "pulled" so far.
        # don't let the value go below 1, so that log() and division still work.
        total_pulls = max(sum(len(r) for r in choice_rewards.values()), 1)

        scores = {}

        for choice, rewards in choice_rewards.items():
            # count the number of pulls for this choice, with a floor of 1
            choice_pulls = max(len(rewards), 1)

            # compute the 2-stdev error for the estimate of this choice
            error = np.sqrt(2.0 * np.log(total_pulls) / choice_pulls)

            # compute the average reward, or default to 0
            avg_reward = np.mean(rewards) if len(rewards) else 0

            # this choice's score is the upper bound of what we think is possible
            scores[choice] = avg_reward + error

        # there may be many choices which are equally desirable
        best_score = max(scores.values())
        best_choices = [k for k, v in scores.items() if v == best_score]

        # make sure we don't choose the same one every time
        return random.choice(best_choices)
