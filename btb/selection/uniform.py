import random

from btb.selection.selector import Selector


class Uniform(Selector):
    """Uniform selector

    Selects a choice uniformly at random.
    """

    def select(self, choice_scores):
        return random.choice(self.choices)
