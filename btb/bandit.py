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
    # total_pulls is capped below at 1, so that log() will not fail
    total_pulls = max(sum(len(r) for r in arms.values()), 1)
    scores = {}
    choices = arms.items()
    choices.shuffle()
    for choice, rewards in arms.items():
        choice_pulls = max(len(rewards), 1)
        error = math.sqrt(2.0 * math.log(total_pulls) / choice_pulls)
        scores[choice] = np.mean(rewards) + error

    best_choice = sorted(scores.keys(), key=scores.get)[-1]
    return best_choice
