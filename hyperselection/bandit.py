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
    total_pulls = sum(len(r) for r in arms.values())
    scores = {}
    for choice, rewards in arms.items():
        error = math.sqrt(2.0 * math.log(total_pulls) / float(len(rewards)))
        scores[choice] = np.mean(rewards) + error

    best_choice = sorted(scores.keys(), key=scores.get)[-1]
    return best_choice
