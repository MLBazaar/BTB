# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np

from btb.tuning.hyperparams.categorical import CategoricalHyperParam


def assert_called_with_np_array(mock_calls, real_calls):
    assert len(mock_calls) == len(real_calls)

    for mock_call, real_call in zip(mock_calls, real_calls):
        np.testing.assert_array_equal(mock_call[0], real_call[1])


class TestCategoricalHyperParam(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.instance = CategoricalHyperParam(choices=['Cat', 'Dog', 'Horse', 'Tiger'])

    @patch('btb.tuning.hyperparams.categorical.OneHotEncoder')
    def test___init__(self, mock_one_hot_encoder):
        """Test that during instantiation we create a OneHotEncoder and we fit it with the given
        choices."""
        # setup
        choices = ['cat', 'dog', 'parrot']
        encoder_instance = MagicMock()
        mock_one_hot_encoder.return_value = encoder_instance

        # run
        instance = CategoricalHyperParam(choices=choices)

        # assert
        self.assertEqual(instance.choices, choices)
        self.assertEqual(instance.default, 'cat')

        # TODO: Fix / reimplmement assert_called_with_np
        # expected_encoder_calls = [
        #     call(categories=[np.array(['cat', 'dog', 'parrot'], dtype=object)], sparse=True)
        # ]

        # assert_called_with_np_array(
        #     mock_one_hot_encoder.call_args_list,
        #     expected_encoder_calls
        # )
        expected_encoder_fit_call = [call(np.array(choices).reshape(-1, 1))]
        assert_called_with_np_array(encoder_instance.fit.call_args_list, expected_encoder_fit_call)

    def test__within_hyperparam_space_values_in_space(self):
        """Test that when we call ``_within_hyperparam_space`` with values in the hyperparameter
        search space does not raise an exception."""
        # setup
        values = ['Cat']
        values_2 = ['Dog', 'Horse']

        # run
        self.instance._within_hyperparam_space(values)
        self.instance._within_hyperparam_space(values_2)

    def test__within_hyperparam_space_values_out_of_space(self):
        """Test that when we call ``_within_hyperparam_space`` with values out of the
        hyperparameter search space does not raise an exception."""
        # setup
        values = ['mat']
        values_2 = ['pug', 'Horse']

        # run/assert
        with self.assertRaises(ValueError):
            self.instance._within_hyperparam_space(values)

        with self.assertRaises(ValueError):
            self.instance._within_hyperparam_space(values_2)

    def test__transform_single_value(self):
        """Test that the method ``transform`` performs a normalization over categorical single
        value and converts it in to a search space of [0, 1]^k.
        """

        # setup
        value = np.array([['Cat']])

        # run
        result = self.instance.transform(value)

        # assert
        expected_result = np.array([[1, 0, 0, 0]])

        np.testing.assert_array_equal(result, expected_result)

    def test__transform_multiple_values(self):
        """Test that the method ``transform`` performs a normalization over categorical single
        value and converts it in to a search space of [0, 1]^k.
        """

        # setup
        values = np.array([['Cat'], ['Dog']])

        # run
        results = self.instance.transform(values)

        # assert
        expected_results = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        np.testing.assert_array_equal(results, expected_results)

    def test__transform_invalid_value(self):
        """Test that a ``ValueError`` is being raised when a value out of the hyperparameter
        space is given.
        """
        # run / assert
        with self.assertRaises(ValueError):
            self.instance._transform(np.array([['cow']]))

    def test__inverse_transform_single_value(self):
        """Test that the method ``_inverse_transform`` performs a denormalization over the search
        space value to the original categorical hyperparameter space.
        """
        # setup
        value = np.array([[1, 0, 0, 0]])

        # run
        result = self.instance.inverse_transform(value)

        # assert
        np.testing.assert_array_equal(result, np.array([['Cat']]))

    def test__inverse_transform_multiple_values(self):
        """Test that the method ``_inverse_transform`` performs a denormalization over the search
        space values to the original categorical hyperparameter space.
        """
        # setup
        values = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # run
        results = self.instance.inverse_transform(values)

        # assert
        np.testing.assert_array_equal(results, np.array([['Cat'], ['Dog']]))

    @patch('btb.tuning.hyperparams.categorical.np.random.random')
    def test_sample(self, mock_np_random):
        """Test that sample returns values."""
        # setup
        mock_np_random.return_value = np.array([
            [0.5, 0.1, 0.2, 0.6],
            [0.1, 0.9, 0.1, 0.6],
        ])
        n = 2

        # run
        results = self.instance.sample(n)

        # assert
        expected_results = np.array([[0, 0, 0, 1], [0, 1, 0, 0]])

        mock_np_random.assert_called_once_with((n, self.instance.dimensions))

        np.testing.assert_array_equal(results, expected_results)

        self.assertEqual(len(results), n)
