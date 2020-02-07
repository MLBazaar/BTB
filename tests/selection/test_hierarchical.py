from unittest import TestCase
from unittest.mock import patch

from btb.selection.hierarchical import HierarchicalByAlgorithm


class TestHierarchicalByAlgorithm(TestCase):
    # METHOD: __init__(self, choices, **kwargs)
    # VALIDATE:
    #     * attribute values
    # TODO:
    #     * Why not re-use choices instead of creating an extra argument?

    def test___init__(self):

        # Run
        by_algorithm = {
            'DT': frozenset(('DT', 'RF', 'ET')),
            'SVM': frozenset(('LSVC', 'NuSVC')),
        }
        selector = HierarchicalByAlgorithm(choices=None, by_algorithm=by_algorithm)

        # Assert
        assert selector.choices is None
        assert selector.by_algorithm == by_algorithm

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * returned value
    #     * arguments passed to self.bandit
    # NOTES:
    #     * self.bandit will need to be mocked to capture the arguments
    #     * Why is a new object created? Why not set self.choices and use super.select?
    #     * Why are frozensets needed? Why not tuples?
    # TODO:
    #     * "if not condition: continue" should rather be "if condition: do something"

    @patch('btb.selection.hierarchical.HierarchicalByAlgorithm.bandit')
    def test_select(self, bandit_mock):
        """If min length is lt k_min, super().compute_rewards is used.

        In this case, we expect the socres to be returned as they are.
        """

        # Set-up
        by_algorithm = {
            'DT': ('DT', 'RF', 'ET'),    # We use tuples here to
            'SVM': ('LSVC', 'NuSVC'),    # preserve the order.
            'KNN': ('KNN', ),
        }
        selector = HierarchicalByAlgorithm(choices=None, by_algorithm=by_algorithm)

        bandit_mock.return_value = 'SVM'

        # Run
        choice_scores = {
            'DT': [0.7, 0.8, 0.9],
            'RF': [0.94],
            'LSVC': [0.88, 0.95, 0.93],
            'NuSVC': [0.89, 0.91, 0.92],
        }
        best = selector.select(choice_scores)

        # Assert
        assert best == 'LSVC'

        alg_scores = {
            'DT': [0.7, 0.8, 0.9, 0.94],
            'SVM': [0.88, 0.95, 0.93, 0.89, 0.91, 0.92]
        }
        bandit_mock.assert_called_once_with(alg_scores)
