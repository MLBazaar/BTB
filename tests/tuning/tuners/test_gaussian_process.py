# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock

from sklearn.gaussian_process import GaussianProcessRegressor

from btb.tuning.tuners.gaussian_process import GPTuner


class TestGaussianProcessTuner(TestCase):
    """Test GaussianProcessTuner class."""

    def test___init__(self):
        # setup
        instance = MagicMock()

        # run
        instance = GPTuner(MagicMock())

        # assert
        assert instance._MODEL_CLASS == GaussianProcessRegressor
