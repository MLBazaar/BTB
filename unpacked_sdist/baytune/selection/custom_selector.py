import random

from baytune.selection.selector import Selector


class CustomSelector(Selector):
    """Custom selector"""

    def select(self, choice_scores):
        """Select a choice uniformly at random."""
        return self.choices[random.randint(0, len(self.choices) - 1)]
