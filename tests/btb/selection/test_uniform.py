from unittest import TestCase

from mock import patch

from btb.selection.uniform import Uniform


class TestUniform(TestCase):

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * returned values and randint call
    # NOTES:
    #     * randint will need to be mocked

    @patch('btb.selection.uniform.random')
    def test_select(self, random_mock):

        # Set-up
        selector = Uniform(['DT', 'RF', 'SVM'])

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
