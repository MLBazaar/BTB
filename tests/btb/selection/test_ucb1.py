from collections import OrderedDict
from unittest import TestCase

from mock import patch

from btb.selection.ucb1 import UCB1


class TestUCB1(TestCase):

    # METHOD: bandit(self, choice_rewards)
    # VALIDATE:
    #     * returned balues
    # NOTES:
    #     * random.choice will need to be mocked

    @patch('btb.selection.ucb1.random')
    def test_bandit(self, random_mock):
        """Only the choices with the highest scores are returned."""

        # Set-up
        selector = UCB1(['DT', 'RF', 'SVM'])

        random_mock.choice.return_value = 'SVM'

        # Run
        choice_rewards = OrderedDict((
            ('DT', [0.7, 0.8, 0.9]),
            ('RF', [0.9, 0.93, 0.95]),
            ('SVM', [0.9, 0.93, 0.95])
        ))

        # We patch dict as OrderedDict to preserve the order
        # in .items() and make the later assert simpler.
        # Otherwise, we could not rely on the list order.
        with patch('btb.selection.ucb1.dict', new=OrderedDict):
            best = selector.bandit(choice_rewards)

        # Assert
        assert best == 'SVM'
        random_mock.choice.assert_called_once_with(['RF', 'SVM'])
