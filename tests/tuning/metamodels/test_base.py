# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from btb.tuning.metamodels.base import BaseMetaModel


class TestBaseMetaModel(TestCase):
    """Test BaseMetaModel class."""

    def test__init_model__MODEL_KWARGS_DEFAULT_none(self):
        # setup
        instance = MagicMock()
        instance._MODEL_KWARGS_DEFAULT = None
        instance._MODEL_CLASS.return_value = 'model_instance'

        # run
        BaseMetaModel._init_model(instance)

        # assert
        assert instance._model_instance == 'model_instance'
        instance._MODEL_CLASS.assert_called_once_with()

    def test__init_model__MODEL_KWARGS_DEFAULT(self):
        # setup
        instance = MagicMock()
        instance._MODEL_KWARGS_DEFAULT = {'a': 1}
        instance._MODEL_CLASS.return_value = 'model_instance'

        # run
        BaseMetaModel._init_model(instance)

        # assert
        assert instance._model_instance == 'model_instance'
        instance._MODEL_CLASS.assert_called_once_with(a=1)

    def test__init_model__self_model_kwargs(self):
        # setup
        instance = MagicMock()
        instance._model_kwargs = {'b': 2}
        instance._MODEL_KWARGS_DEFAULT = {'a': 1}
        instance._MODEL_CLASS.return_value = 'model_instance'

        # run
        BaseMetaModel._init_model(instance)

        # assert
        assert instance._model_instance == 'model_instance'
        instance._MODEL_CLASS.assert_called_once_with(a=1, b=2)

    def test__fit(self):
        # setup
        instance = MagicMock()

        # run
        BaseMetaModel._fit(instance, 'trials', 'scores')

        # assert
        instance._init_model.called_once_with()
        instance._model_instance.fit.called_once_with('trials', 'scores')

    def test__predict(self):
        # setup
        instance = MagicMock()
        instance._model_instance.predict.return_value = [1]

        # run
        result = BaseMetaModel._predict(instance, 'trials')

        # assert
        np.testing.assert_array_equal(result, np.array([1]))
        instance._model_instance.predict.assert_called_once_with('trials')
