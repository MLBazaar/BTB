# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock

from btb.tuning.tuners.uniform import UniformTuner


class TestUniformTuner(TestCase):
    """Test UniformTuner class."""

    def setUp(self):
        self.instance = UniformTuner(MagicMock())

    def test___init__(self):
        assert isinstance(self.instance.tunable, MagicMock)

    def test__propose(self):
        # setup
        self.instance.tunable.sample.return_value = [1]

        # run
        result = self.instance._propose(1)

        # assert
        assert result == [1]
        self.instance.tunable.sample.assert_called_once_with(1)
