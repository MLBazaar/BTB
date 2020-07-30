# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock

from sklearn.gaussian_process import GaussianProcessRegressor

from btb.tuning.tunable import Tunable
from btb.tuning.tuners.gaussian_process import GCPEiTuner, GCPTuner, GPEiTuner, GPTuner


class TestGaussianProcessTuner(TestCase):
    """Test GaussianProcessTuner class."""

    def test___init__(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)
        # run
        instance = GPTuner(tunable)

        # assert
        assert instance._MODEL_CLASS == GaussianProcessRegressor
        assert instance._metamodel_kwargs == {'length_scale': 0.1}

    def test___repr__(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)
        tunable.__str__.return_value = "'tunable'"
        instance = GPTuner(tunable)

        # run
        result = instance.__repr__()

        # assert
        assert result == ("GPTuner(tunable='tunable', maximize=True, num_candidates=1000, "
                          "min_trials=5, length_scale=0.1)")


class TestGaussianProcessExpectedImprovementTuner(TestCase):
    """Test GaussianProcessExpectedImprovementTuner class."""

    def test___init__(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)

        # run
        instance = GPEiTuner(tunable)

        # assert
        assert instance._MODEL_CLASS == GaussianProcessRegressor
        assert instance._metamodel_kwargs == {'length_scale': 0.1}

    def test___repr__(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)
        tunable.__str__.return_value = "'tunable'"
        instance = GPEiTuner(tunable)

        # run
        result = instance.__repr__()

        # assert
        assert result == ("GPEiTuner(tunable='tunable', maximize=True, num_candidates=1000, "
                          "min_trials=5, length_scale=0.1)")


class TestGaussianCopulaProcessExpectedImprovementTuner(TestCase):
    """Test GaussianCopulaProcessExpectedImprovementTuner class."""

    def test___init__(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)

        # run
        instance = GCPEiTuner(tunable)

        # assert
        assert instance._MODEL_CLASS == GaussianProcessRegressor
        assert instance._metamodel_kwargs == {'length_scale': 0.1}

    def test___repr__(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)
        tunable.__str__.return_value = "'tunable'"
        instance = GCPEiTuner(tunable)

        # run
        result = instance.__repr__()

        # assert
        assert result == ("GCPEiTuner(tunable='tunable', maximize=True, num_candidates=1000, "
                          "min_trials=5, length_scale=0.1)")


class TestGaussianCopulaProcessTuner(TestCase):
    """Test GaussianCopulaProcessTuner class."""

    def test___init__(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)
        # run
        instance = GCPTuner(tunable)

        # assert
        assert instance._MODEL_CLASS == GaussianProcessRegressor
        assert instance._metamodel_kwargs == {'length_scale': 0.1}

    def test___repr__(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)
        tunable.__str__.return_value = "'tunable'"
        instance = GCPTuner(tunable)

        # run
        result = instance.__repr__()

        # assert
        assert result == ("GCPTuner(tunable='tunable', maximize=True, num_candidates=1000, "
                          "min_trials=5, length_scale=0.1)")
