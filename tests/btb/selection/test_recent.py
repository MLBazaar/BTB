from unittest import TestCase


class TestRecentKReward(TestCase):

    # METHOD: __init__(self, choices, **kwargs)
    # VALIDATE:
    #     * attribute values
    # TODO:
    #     * kwargs can be safely removed from the method signature
    #     * 'k' should be made an explicit parameter

    # METHOD: compute_rewards(self, scores)
    # VALIDATE:
    #     * returned values

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * Returned values for multiple cases
    #     * arguments passed to bandit
    # TODO:
    #     * This code can be simplified to be more easy to understand
    pass


class TestRecentKVelocity(TestCase):

    # METHOD: compute_rewards(self, scores)
    # VALIDATE:
    #     * returned values
    pass
