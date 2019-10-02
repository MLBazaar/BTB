# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock

from btb.tuning.tuners.base import BaseTuner


class TestBaseTuner(TestCase):
    """Test BaseTuner class."""

    class Tuner(BaseTuner):
        """Simple BaseTuner child."""
        def _propose(self, num_proposals):
            return num_proposals

    def setUp(self):
        tunable = MagicMock()
        self.instance = self.Tuner(tunable)

    def test___init__(self):
        assert isinstance(self.instance.tunable, MagicMock)

    def test_propose_one_value(self):
        """Test that propose method calls it's child implemented method."""

        # setup
        inverse_return = self.instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1]
        self.instance._propose = MagicMock(return_value=1)

        # run
        result = self.instance.propose(1)

        # assert
        self.instance._propose.assert_called_once_with(1)
        self.instance.tunable.inverse_transform.called_once_with(1)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == 1

    def test_propose_many_values(self):
        """Test that propose method calls it's child implemented method."""

        # setup
        inverse_return = self.instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1, 2]
        self.instance._propose = MagicMock(return_value=2)

        # run
        result = self.instance.propose(2)

        # assert
        self.instance._propose.assert_called_once_with(2)
        self.instance.tunable.inverse_transform.called_once_with(2)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == [1, 2]
