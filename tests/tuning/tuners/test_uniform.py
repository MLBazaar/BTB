# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock

from btb.tuning.tunable import Tunable
from btb.tuning.tuners.uniform import UniformTuner


class TestUniformTuner(TestCase):
    """Test UniformTuner class."""

    def setUp(self):
        tunable = MagicMock(spec_set=Tunable)
        self.instance = UniformTuner(tunable)

    def test___init__(self):
        assert isinstance(self.instance.tunable, MagicMock)

    def test__propose_no_duplicates(self):
        """Test that `_propose` method calls it's parent `_sample` method with a single value
        and `allow_duplicates=False`.
        """
        # setup
        self.instance._sample = MagicMock(return_value=[1])

        # run
        result = self.instance._propose(1, False)

        # assert
        assert result == [1]
        self.instance._sample.assert_called_once_with(1, False)

    def test__propose_allow_duplicates(self):
        """Test that `_propose` method calls it's parent `_sample` method with a single value
        and `allow_duplicates=True`.
        """
        # setup
        self.instance._sample = MagicMock(return_value=[1])

        # run
        result = self.instance._propose(1, True)

        # assert
        assert result == [1]
        self.instance._sample.assert_called_once_with(1, True)
