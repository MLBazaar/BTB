# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from btb.tuning.hyperparams.boolean import BooleanHyperParam


class TestBooleanHyperParam(TestCase):
    """Unit test for the class ``BooleanHyperParam``."""

    @classmethod
    def setUpClass(cls):
        """Instantiate a BooleanHyperParam to be used in all the tests."""
        cls.instance = BooleanHyperParam()

    def test__within_hyperparam_space_values_in_space(self):
        """Test that the method ``_within_hyperparam_space`` does not raise a value error when
        values in the hyperparameter space are given."""
        # setup
        values = np.array([[True], [False]])

        # run assert
        self.instance._within_hyperparam_space(values)

    def test__within_hyperparam_space_values_out_of_space(self):
        """Test that the method ``_within_hyperparam_space`` does not raise a value error when
        values in the hyperparameter space are given."""
        # setup
        values = np.array([[1], [0]])

        # run/assert
        with self.assertRaises(ValueError):
            self.instance._within_hyperparam_space(values)

    def test__inverse_transform_single_value(self):
        """Test the method ``_inverse_transform`` to perform a denomalization over a value from
        the search space [0, 1]^K.
        """
        # setup
        values = np.array([[1]])

        # run
        result = self.instance._inverse_transform(values)

        # assert
        np.testing.assert_array_equal(result, np.array([[True]]))

    def test__inverse_transform_multiple_values(self):
        """Test the method ``_inverse_transform`` to perform a denomalization over multiple values
        from the search space [0, 1]^K.
        """
        # setup
        values = np.array([[1], [0], [0], [1]])

        # run
        result = self.instance._inverse_transform(values)

        # assert
        expected_result = np.array([[True], [False], [False], [True]])

        np.testing.assert_array_equal(result, expected_result)

    def test__transform_single_value(self):
        """Test that the method ``_transform`` performs a normalization over a ``boolean`` value
        and converts it in to a search space of [0, 1]^k.
        """

        # setup
        values = np.array([[True]])

        # run
        result = self.instance._transform(values)

        # assert
        np.testing.assert_array_equal(result, np.array([[1]]))

    def test__transform_multiple_values(self):
        """Test that the method ``_transform`` performs a normalization over ``boolean`` values and
        converts them in to a search space of [0, 1]^k.
        """

        # setup
        values = np.array([[True], [False]])

        # run
        result = self.instance._transform(values)

        # assert
        np.testing.assert_array_equal(result, np.array([[1], [0]]))

    @patch('btb.tuning.hyperparams.boolean.np.random.random')
    def test_sample(self, mock_np_random):
        """Test that the method ``sample`` returns values from the search space and not the
        original hyperparameter space.
        """
        # setup
        n = 4
        mock_np_random.return_value = np.array([[0.1], [0.2], [0.5], [0.99]])

        # run
        result = self.instance.sample(n)

        # assert
        expected_result = np.array([[0], [0], [0], [1]])

        mock_np_random.assert_called_once_with((4, 1))
        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(result, expected_result)
