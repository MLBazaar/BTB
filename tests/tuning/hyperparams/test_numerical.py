# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from btb.tuning.hyperparams.numerical import FloatHyperParam, IntHyperParam


class TestFloatHyperParam(TestCase):

    def setUp(self):
        self.instance = FloatHyperParam(min=0.1, max=0.9)

    def test_transform(self):
        """Test that the method `transform` performs a normalization over categorical values and
        converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = [0.9, 0.8]
        values_2 = [0.1, 0.5]
        values_3 = 0.2

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)
        result_3 = self.instance.transform(values_3)

        # assert
        expected_result_1 = np.array([[1.], [0.875]])
        expected_result_2 = np.array([[0.], [0.5]])
        expected_result_3 = 0.125

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)
        assert result_3 == expected_result_3

    def test_inverse_transform(self):
        """Test that the method `inverse_transform` performs a denormalization over the search
        space values to the original categorical hyperparameter space.
        """
        # setup
        values_1 = [1., 0.875]
        values_2 = 0.125

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)

        # assert
        expected_result_1 = [[0.9], [0.8]]
        expected_result_2 = 0.2

        np.testing.assert_array_equal(result_1, expected_result_1)
        assert result_2 == expected_result_2

    def test_sample(self):
        """Test that the returned sample values are not bigger than 1 and not smaller than 0"""

        # run
        results = self.instance.sample(10)

        # assert
        assert len(results) == 10
        assert all(res < 1 and res > 0 for res in results)


class TestIntHyperParam(TestCase):

    def setUp(self):
        self.instance = IntHyperParam(0, 9)

    def test_transform(self):
        """Test that the method `transform` performs a normalization over categorical values and
        converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = [2, 6]
        values_2 = [7, 0]
        values_3 = 0

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)
        result_3 = self.instance.transform(values_3)

        # assert
        expected_result_1 = np.array([[0.25], [0.65]])
        expected_result_2 = np.array([[0.75], [0.05]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)
        assert result_3 == 0.05

    def test_inverse_transform(self):
        """Test that the method `inverse_transform` performs a denormalization over the search
        space values to the original categorical hyperparameter space.
        """
        # setup
        values_1 = [0.35, 0.65]
        values_2 = 0.05

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)

        # assert
        expected_result_1 = [[3], [6]]
        expected_result_2 = 0

        np.testing.assert_array_equal(result_1, expected_result_1)
        assert result_2 == expected_result_2

    def test_sample(self):
        """Test that the returned sample values are not bigger than 1 and not smaller than 0"""

        # run
        results = self.instance.sample(10)

        # assert
        assert len(results) == 10
        assert all(res < 1 and res > 0 for res in results)
