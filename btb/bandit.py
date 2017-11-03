import math
import random
import numpy as np


def ucb1_bandit(arms):
    """
    Multi-armed bandit which chooses the arm for which the upper confidence
    bound (UCB) of expected reward is greatest.
    An explanation is here:
    https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf

    arms: maps choice IDs to lists of rewards.
        {choice -> list[float]}
    """
    # count the total number of times all "levers" have been "pulled" so far.
    # don't let the value go below 1, so that log() and division still work.
    total_pulls = max(sum(len(r) for r in arms.values()), 1)
    scores = {}

    # shuffle the arms so that if all else is equal, we don't choose the same
    # one every time
    choices = arms.items()
    random.shuffle(choices)

    for choice, rewards in choices:
        # count the number of pulls for this choice, with a floor of 1
        choice_pulls = max(len(rewards), 1)

        # compute the 2-stdev error for the estimate of this choice
        error = math.sqrt(2.0 * math.log(total_pulls) / choice_pulls)

        # this choice's score is the upper bound of what we think is possible
        scores[choice] = np.mean(rewards) + error

    best_choice = sorted(scores.keys(), key=scores.get)[-1]
    return best_choice
