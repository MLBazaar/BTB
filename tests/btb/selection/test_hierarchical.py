from unittest import TestCase


class TestHierarchicalByAlgorithm(TestCase):
    # METHOD: __init__(self, choices, **kwargs)
    # VALIDATE:
    #     * attribute values
    # TODO:
    #     * kwargs can be safely removed from the method signature
    #     * by_algorithm should be made an explicit parameter

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * returned value
    #     * arguments passed to self.bandit
    # NOTES:
    #     * self.bandit will need to be mocked to capture the arguments
    # TODO:
    #     * "if not condition: continue" should rather be "if condition: do something"
    pass
