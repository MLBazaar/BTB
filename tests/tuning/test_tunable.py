# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd

from btb.tuning.tunable import Tunable


def assert_called_with_np_array(mock_calls, real_calls):
    assert len(mock_calls) == len(real_calls)

    for mock_call, real_call in zip(mock_calls, real_calls):
        np.testing.assert_array_equal(mock_call[0], real_call[1])


class TestTunable(TestCase):
    """Unit test for the class ``Tunable``."""

    @classmethod
    def setUpClass(cls):
        """Instantiate the ``Tunable`` and it's ``Hyperparameters`` that we will be using."""
        cls.bhp = MagicMock()
        cls.chp = MagicMock()
        cls.ihp = MagicMock()

        hyperparams = {
            'bhp': cls.bhp,
            'chp': cls.chp,
            'ihp': cls.ihp,
        }

        cls.instance = Tunable(hyperparams, names=['bhp', 'chp', 'ihp'])

    def test___init__(self):
        """Test that the names are being generated correctly."""

        # setup
        expected_names = ['bhp', 'chp', 'ihp']

        # assert
        assert all(name in self.instance.names for name in expected_names)

    def test_transform_valid_values(self):
        """Test transform with valid values only."""
        # setup
        self.bhp.transform.return_value = [[1]]
        self.chp.transform.return_value = [[0]]
        self.ihp.transform.return_value = [[1]]

        values_dict = {'bhp': True, 'chp': 'cat', 'ihp': 1}
        values_list_dict = [
            {'bhp': True, 'chp': 'cat', 'ihp': 2},
            {'bhp': False, 'chp': 'cat', 'ihp': 3}
        ]

        values_series = pd.Series([False, 'cat', 1], index=['bhp', 'chp', 'ihp'])
        values_array = [[True, 'dog', 2]]
        values_larger_array = [[True, 'dog', 2], [False, 'cat', 3]]

        values_df = pd.DataFrame([{'bhp': True, 'chp': 'cat', 'ihp': 1}])

        # run
        self.instance.transform(values_dict)
        self.instance.transform(values_list_dict)
        self.instance.transform(values_series)
        self.instance.transform(values_array)
        self.instance.transform(values_larger_array)
        self.instance.transform(values_df)

        # assert
        expected_call_bhp_transform = [
            call(np.array([True])),
            call(np.array([True, False])),
            call(np.array([False], dtype=object)),
            call(np.array([True])),
            call(np.array([True, False])),
            call(np.array([True]))]

        expected_call_chp_transform = [
            call(np.array(['cat'])),
            call(np.array(['cat', 'cat'])),
            call(np.array(['cat'])),
            call(np.array(['dog'])),
            call(np.array(['dog', 'cat'])),
            call(np.array(['cat'])),
        ]

        expected_call_ihp_transform = [
            call(np.array([1])),
            call(np.array([2, 3])),
            call(np.array([1])),
            call(np.array([2])),
            call(np.array([2, 3])),
            call(np.array([1])),
        ]

        assert_called_with_np_array(self.bhp.transform.call_args_list, expected_call_bhp_transform)
        assert_called_with_np_array(self.chp.transform.call_args_list, expected_call_chp_transform)
        assert_called_with_np_array(self.ihp.transform.call_args_list, expected_call_ihp_transform)

    def test_transform_invalid_values(self):
        """Test transform method with invalid values."""

        # setup
        values_1 = [False, 'cat', 1]
        values_2 = [[False, True], ['cat', 'dog'], [1, 1]]

        # run / assert

        with self.assertRaises(ValueError):
            self.instance.transform(values_1)

        with self.assertRaises(ValueError):
            self.instance.transform(values_2)

        self.bhp.transform.assert_not_called()
        self.chp.transform.assert_not_called()
        self.ihp.transform.assert_not_called()

    def test_inverse_transform(self):
        """Test the inverse transform method is calling the hyperparameters."""
        # setup
        self.bhp.K = 1
        self.chp.K = 1
        self.ihp.K = 1

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
        pd.testing.assert_frame_equal(expected_result, result)

    def test_sample(self):
        """Test that the method sample generates data from all the ``hyperparams``"""

        # setup
        self.bhp.sample.return_value = [[1]]
        self.chp.sample.return_value = [[1, 1]]
        self.ihp.sample.return_value = [[1]]

        # run
        result = self.instance.sample(1)

        # assert
        expected_result = [[1, 1, 1, 1]]

        self.bhp.sample.assert_called_once_with(1)
        self.chp.sample.assert_called_once_with(1)
        self.ihp.sample.assert_called_once_with(1)
        assert (result == expected_result).all()
