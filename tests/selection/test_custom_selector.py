from unittest import TestCase
from unittest.mock import patch

from btb.selection.custom_selector import CustomSelector


class TestCustomSelector(TestCase):

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * returned values and randint call
    # NOTES:
    #     * randint will need to be mocked

    @patch('btb.selection.custom_selector.random')
    def test_select(self, random_mock):

        # Set-up
        selector = CustomSelector(['DT', 'RF', 'SVM'])

        random_mock.randint.return_value = 0

        # Run
        choice_scores = {
            'DT': [0.7, 0.73, 0.75],
            'RF': [0.8, 0.83, 0.85],
            'SVM': [0.9, 0.93, 0.95]
        }
        best = selector.select(choice_scores)

        # Assert
        assert best == 'DT'
        random_mock.randint.assert_called_once_with(0, 2)
