from unittest import TestCase

from btb.selection.selector import Selector


class TestSelector(TestCase):

    # METHOD: __init__(self, choices, **kwargs)
    # VALIDATE:
    #     * attribute value

    def test___init__(self):

        # Run
        selector = Selector(['RF', 'SVM'])

        # Assert
        assert selector.choices == ['RF', 'SVM']

    # METHOD: compute_rewards(self, scores)
    # VALIDATE:
    #     * returned value
    def test_compute_rewards(self):

        # Set-up
        selector = Selector(['RF', 'SVM'])

        # Run
        scores = [0.8, 0.9]
        rewards = selector.compute_rewards(scores)

        # Assert
        assert rewards == scores

    # METHOD: bandit(self, choice_rewards)
    # VALIDATE:
    #     * returned value

    def test_bandit(self):

        # Set-up
        selector = Selector(['RF', 'SVM'])

        # Run
        choice_rewards = {
            'RF': [0.8, 0.83, 0.85],
            'SVM': [0.9, 0.93, 0.95]
        }
        best = selector.bandit(choice_rewards)

        # Assert
        assert best == 'SVM'

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * returned values
    # TODO:
    #     * "if not condition: continue" should rather be "if condition: do something"

    def test_select(self):

        # Set-up
        selector = Selector(['RF', 'SVM'])

        # Run
        choice_scores = {
            'DT': [0.7, 0.73, 0.75],
            'RF': [0.8, 0.83, 0.85],
            'SVM': [0.9, 0.93, 0.95]
        }
        best = selector.select(choice_scores)

        # Assert
        assert best == 'SVM'
