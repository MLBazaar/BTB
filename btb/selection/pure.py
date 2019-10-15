import logging

from btb.selection.selector import Selector

# the minimum number of scores that each choice must have in order to use best-K
# optimizations. If not all choices meet this threshold, default UCB1 selection
# will be used.
K_MIN = 3

logger = logging.getLogger('btb')


class PureBestKVelocity(Selector):
    """Pure Best K Velocity Selector

    Simply returns the choice with the best best-K velocity.
    """

    def __init__(self, choices, k=K_MIN):
        super(PureBestKVelocity, self).__init__(choices)
        self.k = k

    def compute_rewards(self, scores):
        """
        Compute the "velocity" of (average distance between) the k+1 best
        scores. Return a list with those k velocities padded out with zeros so
        that the count remains the same.
        """
        # get the k + 1 best scores in descending order
        best_scores = sorted(scores, reverse=True)[:self.k + 1]
        velocities = [best_scores[i] - best_scores[i + 1]
                      for i in range(len(best_scores) - 1)]

        # pad the list out with zeros to maintain the length of the list
        zeros = (len(scores) - self.k) * [0]
        return velocities + zeros

    def select(self, choice_scores):
        """
        Select the choice with the highest best-K velocity. If any choices
        don't have MIN_K scores yet, return the one with the fewest.
        """
        # if we don't have enough scores to do K-selection, fall back to UCB1
        min_num_scores = min([len(s) for s in choice_scores.values()])
        if min_num_scores >= K_MIN:
            logger.info('PureBestKVelocity: using Pure Best K velocity selection')
            reward_func = self.compute_rewards
        else:
            logger.warning(
                '{klass}: Not enough choices to do K-selection; '
                'returning choice with fewest scores'
                .format(klass=type(self).__name__))
            # reward choices with the fewest scores
            # NOTE: "reward_func = lambda " changed to "def reward_func"
            # as per flake8 suggestions
            # reward_func = lambda s: [1] if len(s) == min_num_scores else [0]

            def reward_func(scores):
                return [1] if len(scores) == min_num_scores else [0]

        choice_rewards = {}
        for choice, scores in choice_scores.items():
            if choice not in self.choices:
                continue
            choice_rewards[choice] = reward_func(scores)

        # the default bandit returns the choice with the highest mean reward
        return self.bandit(choice_rewards)
