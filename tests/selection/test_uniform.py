from unittest import TestCase
from unittest.mock import patch

from btb.selection.uniform import Uniform


class TestUniform(TestCase):

    # METHOD: select(self, choice_scores)
    # VALIDATE:
    #     * returned values and randint call
    # NOTES:
    #     * randint will need to be mocked

    @patch('random.choice')
    def test_select(self, mock_choice):

        # Set-up
        choices = ['DT', 'RF', 'SVM']
        selector = Uniform(choices)
        expected = 'DT'

        mock_choice.return_value = expected

        # Run
        choice_scores = None
        actual = selector.select(choice_scores)

        # Assert
        assert actual == expected
        mock_choice.assert_called_once_with(choices)
