# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from btb.tuning.hyperparams.categorical import CategoricalHyperParam


class TestCategoricalHyperParam(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.instance = CategoricalHyperParam(choices=['Cat', 'Dog', 'Horse', 'Tiger'])

    def test_transform_list_values(self):
        """Test that the method ``transform`` performs a normalization over categorical list of
        values and converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = ['Cat', 'Dog', 'Horse']
        values_2 = [['Tiger'], ['Cat']]

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)

        # assert
        expected_result_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        expected_result_2 = np.array([[0, 0, 0, 1], [1, 0, 0, 0]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)

    def test_transform_single_value(self):
        """Test that the method ``transform`` performs a normalization over a categorical value and
        converts it in to a search space of [0, 1]^k.
        """
        # run
        result_1 = self.instance.transform('Tiger')
        result_2 = self.instance.transform('Cat')

        # assert
        expected_result_1 = np.array([[0, 0, 0, 1]])
        expected_result_2 = np.array([[1, 0, 0, 0]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)

    def test_transform_invalid_value(self):
        """Test that a ``ValueError`` is being raised when a value out of the hyperparameter
        space is given.
        """
        # run / assert
        with self.assertRaises(ValueError):
            self.instance.transform('cow')

    def test_inverse_transform(self):
        """Test that the method `inverse_transform` performs a denormalization over the search
        space values to the original categorical hyperparameter space.
        """
        # setup
        values_1 = np.array([[1, 0, 0, 0]])
        values_2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)

        # assert
        self.assertEqual(result_1, 'Cat')
        np.testing.assert_array_equal(result_2, [['Cat'], ['Dog']])

    def test_sample(self):
        """Test that sample returns values."""
        # setup
        n = 10

        # run
        results = self.instance.sample(n)

        # assert
        self.assertEqual(len(results), n)
