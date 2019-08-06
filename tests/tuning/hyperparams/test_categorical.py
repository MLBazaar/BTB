# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from btb.tuning.hyperparams.categorical import CategoricalHyperParam


class TestCategoricalHyperParam(TestCase):

    def setUp(self):
        self.instance = CategoricalHyperParam(['Cat', 'Dog', 'Horse', 'Tiger'])

    def test_transform_list(self):
        """Test that the method ``transform`` performs a normalization over categorical list of
        values and converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = ['Cat', 'Dog', 'Horse']
        values_2 = ['Tiger', 'Cat']

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)

        # assert
        expected_result_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        expected_result_2 = np.array([[0, 0, 0, 1], [1, 0, 0, 0]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)

    def test_transform_scalar(self):
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

    def test_inverse_transform(self):
        """Test that the method `inverse_transform` performs a denormalization over the search
        space values to the original categorical hyperparameter space.
        """
        # setup
        values_1 = np.array([1, 0, 0, 0])
        values_2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)

        # assert
        assert result_1 == 'Cat'
        np.testing.assert_array_equal(result_2, [['Cat'], ['Dog']])

    def test_sample(self):
        """Test that sample returns values."""

        # run
        results = self.instance.sample(10)

        # assert
        assert len(results) == 10
        assert all(isinstance(res[0], int) for res in results.tolist())
