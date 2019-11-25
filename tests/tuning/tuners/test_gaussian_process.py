# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock

from sklearn.gaussian_process import GaussianProcessRegressor

from btb.tuning.tuners.gaussian_process import GPEiTuner, GPTuner


class TestGaussianProcessTuner(TestCase):
    """Test GaussianProcessTuner class."""

    def test___init__(self):
        # setup
        instance = MagicMock()

        # run
        instance = GPTuner(MagicMock())

        # assert
        assert instance._MODEL_CLASS == GaussianProcessRegressor
        assert instance._metamodel_kwargs == {'length_scale': 0.1}


class TestGaussianProcessExpectedImprovementTuner(TestCase):
    """Test GaussianProcessExpectedImprovementTuner class."""

    def test___init__(self):
        # setup
        instance = MagicMock()

        # run
        instance = GPEiTuner(MagicMock())

        # assert
        assert instance._MODEL_CLASS == GaussianProcessRegressor
        assert instance._metamodel_kwargs == {'length_scale': 0.1}
