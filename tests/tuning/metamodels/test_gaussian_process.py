# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from btb.tuning.metamodels.gaussian_process import GaussianProcessMetaModel


class TestGaussianProcessMetaModel(TestCase):

    def test___init__(self):
        # run
        instance = GaussianProcessMetaModel()

        # assert
        assert instance._MODEL_KWARGS_DEFAULT == {'normalize_y': True}
        assert instance._MODEL_CLASS == GaussianProcessRegressor

    def test__predict(self):
        # setup
        instance = MagicMock()
        instance._model_instance.predict.return_value = [[1], [2]]

        # run
        result = GaussianProcessMetaModel._predict(instance, 1)

        # assert
        instance._model_instance.predict.assert_called_once_with(1, return_std=True)
        np.testing.assert_array_equal(result, np.array([[1, 2]]))
