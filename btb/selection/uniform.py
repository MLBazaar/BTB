import random

from btb.selection.selector import Selector


class Uniform(Selector):
    """Uniform selector

    Selects a choice uniformly at random.
    """

    def select(self, choice_scores):
        return self.choices[random.randint(0, len(self.choices) - 1)]
