# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd

from btb.tuning.hyperparams.boolean import BooleanHyperParam
from btb.tuning.hyperparams.categorical import CategoricalHyperParam
from btb.tuning.hyperparams.numerical import IntHyperParam
from btb.tuning.tunable import Tunable


def assert_called_with_np_array(mock_calls, real_calls):
    assert len(mock_calls) == len(real_calls)

    for mock_call, real_call in zip(mock_calls, real_calls):
        np.testing.assert_array_equal(mock_call[0], real_call[1])


class TestTunable(TestCase):
    """Unit test for the class ``Tunable``."""

    @patch('btb.tuning.tunable.list')
    def setUp(self, list_mock):
        """Instantiate the ``Tunable`` and it's ``Hyperparameters`` that we will be using."""
        self.bhp = MagicMock(spec_set=BooleanHyperParam)
        self.chp = MagicMock(spec_set=CategoricalHyperParam)
        self.ihp = MagicMock(spec_set=IntHyperParam)

        list_mock.return_value = ['bhp', 'chp', 'ihp']

        hyperparams = {
            'bhp': self.bhp,
            'chp': self.chp,
            'ihp': self.ihp,
        }

        self.instance = Tunable(hyperparams)

    def test___init__with_given_names(self):
        """Test that the names are being generated correctly."""
        # assert
        assert self.instance.names == ['bhp', 'chp', 'ihp']

    def test_transform_valid_dict(self):
        """Test transform method with a dictionary that has all the hyperparameters."""
        # setup
        self.bhp.transform.return_value = [[1]]
        self.chp.transform.return_value = [[0]]
        self.ihp.transform.return_value = [[1]]

        values_dict = {
            'bhp': True,
            'chp': 'cat',
            'ihp': 1
        }

        # run
        result = self.instance.transform(values_dict)

        # assert
        self.bhp.transform.assert_called_once_with(True)
        self.chp.transform.assert_called_once_with('cat')
        self.ihp.transform.assert_called_once_with(1)

        np.testing.assert_array_equal(result, np.array([[1, 0, 1]]))

    def test_transform_empty_dict(self):
        """Test transform method with a dictionary that has a missing hyperparameters."""
        # run / assert
        with self.assertRaises(KeyError):
            self.instance.transform({})

    def test_transform_invalid_dict_one_missing(self):
        """Test transform method with a dictionary that has a missing hyperparameters."""
        # run / assert
        values = {
            'bhp': True,
            'chp': 'cat'
        }

        with self.assertRaises(KeyError):
            self.instance.transform(values)

    def test_transform_list_of_dicts(self):
        """Test transform method with a list of dictionaries."""
        # setup
        self.bhp.transform.return_value = [[1], [0]]
        self.chp.transform.return_value = [[0], [1]]
        self.ihp.transform.return_value = [[1], [1]]

        values_list_dict = [
            {'bhp': True, 'chp': 'cat', 'ihp': 2},
            {'bhp': False, 'chp': 'cat', 'ihp': 3}
        ]

        # run
        results = self.instance.transform(values_list_dict)

        # assert
        assert_called_with_np_array(self.bhp.transform.call_args_list, [call([True, False])])
        assert_called_with_np_array(self.chp.transform.call_args_list, [call(['cat', 'cat'])])
        assert_called_with_np_array(self.ihp.transform.call_args_list, [call([2, 3])])

        np.testing.assert_array_equal(results, np.array([[1, 0, 1], [0, 1, 1]]))

    def test_transform_list_of_invalid_dicts(self):
        """Test transform method with a list of dictionaries where one of them does not have
        the categorical value."""

        # setup
        self.bhp.transform.return_value = [[1], [0]]

        # Here we create a CHP so we can raise an value error as there will be a NaN inside the
        # pandas.DataFrame.
        self.chp = CategoricalHyperParam(['cat', 'dog'])
        self.ihp.transform.return_value = [[1], [1]]

        values_list_dict = [
            {'bhp': True, 'ihp': 2},
            {'bhp': False, 'chp': 'cat', 'ihp': 3}
        ]

        # run / assert
        with self.assertRaises(ValueError):
            self.instance.transform(values_list_dict)

    def test_transform_empty_list(self):
        """Test transform method with an empty list."""
        # run / assert
        with self.assertRaises(IndexError):
            self.instance.transform(list())

    def test_transform_valid_pandas_series(self):
        """Test transform method over a valid ``pandas.Series`` object."""
        # setup
        self.bhp.transform.return_value = [[1]]
        self.chp.transform.return_value = [[0]]
        self.ihp.transform.return_value = [[1]]

        values = pd.Series([False, 'cat', 1], index=['bhp', 'chp', 'ihp'])

        # run
        result = self.instance.transform(values)

        # assert
        self.bhp.transform.assert_called_once_with(False)
        self.chp.transform.assert_called_once_with('cat')
        self.ihp.transform.assert_called_once_with(1)

        np.testing.assert_array_equal(result, np.array([[1, 0, 1]]))

    def test_transform_invalid_pandas_series(self):
        """Test transform method over a ``pandas.Series`` object that does not have index."""
        # setup
        values = pd.Series([False, 'cat', 1])

        # run
        with self.assertRaises(KeyError):
            self.instance.transform(values)

    def test_transform_array_like_list(self):
        """Test transform a valid array like list."""
        # setup
        self.bhp.transform.return_value = [[1]]
        self.chp.transform.return_value = [[0]]
        self.ihp.transform.return_value = [[1]]

        values = [[True, 'dog', 2], [False, 'cat', 3]]

        # run
        result = self.instance.transform(values)

        # assert
        assert_called_with_np_array(
            self.bhp.transform.call_args_list,
            [call(np.array([True, False]))]
        )
        assert_called_with_np_array(
            self.chp.transform.call_args_list,
            [call(np.array(['dog', 'cat']))]
        )
        assert_called_with_np_array(
            self.ihp.transform.call_args_list,
            [call(np.array([2, 3]))]
        )

        np.testing.assert_array_equal(result, np.array([[1, 0, 1]]))

    def test_transform_simple_list(self):
        """Test that the method transform performs a transformation over a list with a single
        combination of hyperparameter valid values.
        """
        # setup
        self.bhp.transform.return_value = [[1]]
        self.chp.transform.return_value = [[0]]
        self.ihp.transform.return_value = [[1]]

        values = [True, 'dog', 2]

        # run
        result = self.instance.transform(values)

        # assert
        self.bhp.transform.assert_called_once_with(True)
        self.chp.transform.assert_called_once_with('dog')
        self.ihp.transform.assert_called_once_with(2)

        np.testing.assert_array_equal(result, np.array([[1, 0, 1]]))

    def test_transform_pd_df(self):
        """Test that the method transform performs a transformation over a ``pandas.DataFrame``
        with a single combination of hyperparameter valid values.
        """
        # setup
        self.bhp.transform.return_value = [[1]]
        self.chp.transform.return_value = [[0]]
        self.ihp.transform.return_value = [[1]]

        values = pd.DataFrame([[True, 'dog', 2]], columns=['bhp', 'chp', 'ihp'])

        # run
        result = self.instance.transform(values)

        # assert
        self.bhp.transform.assert_called_once_with(True)
        self.chp.transform.assert_called_once_with('dog')
        self.ihp.transform.assert_called_once_with(2)

        np.testing.assert_array_equal(result, np.array([[1, 0, 1]]))

    def test_transform_simple_invalid_list(self):
        """Test that the method transform does not transform a list with a single combination
        of invalid hyperparameter values.
        """
        # run / assert
        with self.assertRaises(TypeError):
            self.instance.transform([[True], 1, 2])

    def test_inverse_transform_valid_data(self):
        """Test that the inverse transform method is calling the hyperparameters."""
        # setup
        self.bhp.inverse_transform.return_value = [[True]]
        self.chp.inverse_transform.return_value = [['cat']]
        self.ihp.inverse_transform.return_value = [[1]]

        values = [[1, 0, 1]]

        # run
        result = self.instance.inverse_transform(values)

        # assert
        expected_result = pd.DataFrame(
            {
                'bhp': [True],
                'chp': ['cat'],
                'ihp': [1]
            },
            dtype=object
        )

        self.bhp.inverse_transform.assert_called_once_with([1])
        self.chp.inverse_transform.assert_called_once_with([0])
        self.ihp.inverse_transform.assert_called_once_with([1])
        pd.testing.assert_frame_equal(result, expected_result)

    def test_inverse_transform_invalid_data(self):
        """Test that the a ``TypeError`` is being raised when calling with the invalid data."""
        # setup
        values = [1, 0, 1]

        # run
        with self.assertRaises(TypeError):
            self.instance.inverse_transform(values)

    def test_sample(self):
        """Test that the method sample generates data from all the ``hyperparams``."""

        # setup
        # Values have been changed to ensure that each one of them is being called.
        self.bhp.sample.return_value = [['a']]
        self.chp.sample.return_value = [['b']]
        self.ihp.sample.return_value = [['c']]

        # run
        result = self.instance.sample(1)

        # assert
        expected_result = np.array([['a', 'b', 'c']])

        assert set(result.flat) == set(expected_result.flat)
        self.bhp.sample.assert_called_once_with(1)
        self.chp.sample.assert_called_once_with(1)
        self.ihp.sample.assert_called_once_with(1)

    def test_get_defaults(self):
        # setup
        bhp = MagicMock(default=True)
        chp = MagicMock(default='test')
        ihp = MagicMock(default=1)

        hyperparams = {
            'bhp': bhp,
            'chp': chp,
            'ihp': ihp,
        }

        self.instance = Tunable(hyperparams)

        # run
        result = self.instance.get_defaults()

        # assert
        assert result == {'bhp': True, 'chp': 'test', 'ihp': 1}

    def test_from_dict_not_a_dict(self):
        # run
        with self.assertRaises(TypeError):
            Tunable.from_dict(1)

    @patch('btb.tuning.tunable.IntHyperParam')
    @patch('btb.tuning.tunable.FloatHyperParam')
    @patch('btb.tuning.tunable.CategoricalHyperParam')
    @patch('btb.tuning.tunable.BooleanHyperParam')
    def test_from_dict(self, mock_bool, mock_cat, mock_float, mock_int):
        # setup
        mock_bool.return_value.dimensions = 1
        mock_cat.return_value .dimensions = 1
        mock_float.return_value.dimensions = 1
        mock_int.return_value.dimensions = 1

        mock_bool.return_value.cardinality = 1
        mock_cat.return_value .cardinality = 1
        mock_float.return_value.cardinality = 1
        mock_int.return_value.cardinality = 1

        # run
        hyperparameters = {
            'bhp': {
                'type': 'bool',
                'default': False
            },
            'chp': {
                'type': 'str',
                'default': 'cat',
                'range': ['a', 'b', 'cat']
            },
            'fhp': {
                'type': 'float',
                'default': None,
                'range': [0.1, 1.0]
            },
            'ihp': {
                'type': 'int',
                'default': 5,
                'range': [1, 10]
            }
        }

        result = Tunable.from_dict(hyperparameters)

        # assert
        mock_bool.assert_called_once_with(default=False)
        mock_cat.assert_called_once_with(choices=['a', 'b', 'cat'], default='cat')
        mock_float.assert_called_once_with(min=0.1, max=1.0, default=None)
        mock_int.assert_called_once_with(min=1, max=10, default=5)

        expected_tunable_hp = {
            'bhp': mock_bool.return_value,
            'chp': mock_cat.return_value,
            'fhp': mock_float.return_value,
            'ihp': mock_int.return_value
        }

        assert result.hyperparams == expected_tunable_hp
        assert result.dimensions == 4
        assert result.cardinality == 1
