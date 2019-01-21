import numpy as np


class Selector(object):
    """Base selector

    Args:
        choices (list): a list of discrete choices from which the selector must choose at every
            call to ``select``.
    """

    def __init__(self, choices):
        self.choices = choices

    def compute_rewards(self, scores):
        """Compute rewards from choice's scores

        Convert a list of scores associated with one choice into a list of rewards. Normally, the
        length of the list will be preserved, even if some of the scores are dropped.
        """
        return list(scores)

    def bandit(self, choice_rewards):
        """Return the choice to take next using multi-armed bandit

        Multi-armed bandit method. Accepts a mapping of choices to rewards which indicate their
        historical performance, and returns the choice that we should make next in order to
        maximize expected reward in the long term.

        The default implementation is to return the arm with the highest average score.

        Args:
            choice_rewards (Dict[object, List[float]]): maps choice IDs to lists of rewards.

        Returns:
            str: the name of the choice to take next.
        """
        return max(choice_rewards, key=lambda a: np.mean(choice_rewards[a]))

    def select(self, choice_scores):
        """Select the next best choice to make

        Args:
            choice_scores (Dict[object, List[float]]): Mapping of choice to list of scores for each
                possible choice. The caller is responsible for making sure each choice that is
                possible at this juncture is represented in the dict, even those with no scores.
                Score lists should be in ascending chronological order, that is, the score from the
                earliest trial should be listed first.

                For example::

                    {
                        1: [0.56, 0.61, 0.33, 0.67],
                        2: [0.25, 0.58],
                        3: [0.60, 0.65, 0.68],
                    }
        """
        choice_rewards = {}
        for choice, scores in choice_scores.items():
            if choice not in self.choices:
                continue

            choice_rewards[choice] = self.compute_rewards(scores)

        return self.bandit(choice_rewards)
