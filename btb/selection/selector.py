from builtins import object
import random


class Selector(object):
    def __init__(self, choices, **kwargs):
        """
        Args:
            choices: a list of discrete choices from which the selector must
                choose at every call to select().
        """
        self.choices = choices

    def compute_rewards(self, scores):
        """
        Convert a list of scores associated with one choice into a list of
        rewards. Normally, the length of the list will be preserved, even if
        some of the scores are dropped.
        """
        return list(scores)

    def bandit(self, choice_rewards):
        """
        Multi-armed bandit method. Accepts a mapping of choices to rewards which
        indicate their historical performance, and returns the choice that we
        should make next in order to maximize expected reward in the long term.

        Args:
            choice_rewards: maps choice IDs to lists of rewards.
                {choice -> [float]}

        Returns:
            choice: string indicating the name of the choice to take next.
        """
        # default implementation: return the arm with the highest average score
        return max(choice_rewards, key=lambda a: np.mean(choice_rewards[a]))

    def select(self, choice_scores):
        """
        Select the next best choice to make

        Args:
            choice_scores: map of {choice -> [scores]} for each possible choice. The
                caller is responsible for making sure each choice that is possible at
                this juncture is represented in the dict, even those with no scores.
                Score lists should be in ascending chronological order.
                e.g.
                {1: [0.56, 0.61, 0.33, 0.67],
                 2: [0.25, 0.58],
                 3: [0.60, 0.65, 0.68]}
        """
        choice_rewards = {}
        for choice, scores in choice_scores.items():
            if choice not in self.choices:
                continue
            choice_rewards[choice] = self.compute_rewards(scores)

        return self.bandit(choice_rewards)
