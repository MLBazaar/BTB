# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from btb.tuning.hyperparams.numerical import FloatHyperParam, IntHyperParam


class TestFloatHyperParam(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.instance = FloatHyperParam(min=0.1, max=0.9)

    def test_transform(self):
        """Test that the method `transform` performs a normalization over categorical values and
        converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = [[0.9], [0.8]]
        values_2 = [0.1]
        values_3 = 0.2

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)
        result_3 = self.instance.transform(values_3)

        # assert
        expected_result_1 = np.array([[1.], [0.875]])
        expected_result_2 = np.array([[0.]])
        expected_result_3 = np.array([[0.125]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)
        self.assertEqual(result_3, expected_result_3)

    def test_transform_invalid_range(self):

        # run / assert
        with self.assertRaises(ValueError):
            self.instance.transform(1)

        with self.assertRaises(ValueError):
            self.instance.transform([0.1, 1])

    def test_inverse_transform(self):
        """Test that the method `inverse_transform` performs a denormalization over the search
        space values to the original categorical hyperparameter space.
        """
        # setup
        values_1 = [[0.25], [0.375]]
        values_2 = [1., 0.875]
        values_3 = 0.125

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)
        result_3 = self.instance.inverse_transform(values_3)

        # assert
        expected_result_1 = np.array([[0.3], [0.4]])
        expected_result_2 = np.array([[0.9], [0.8]])
        expected_result_3 = np.array([[0.2]])

        np.testing.assert_almost_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)
        self.assertEqual(result_3, expected_result_3)

    def test_sample(self):
        """Test that the returned sample values are not bigger than 1 and not smaller than 0"""

        # setup
        n = 10

        # run
        results = self.instance.sample(n)

        # assert
        self.assertEqual(len(results), n)
        assert all(0 < res and res < 1 for res in results)


class TestIntHyperParam(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.instance = IntHyperParam(min=0, max=9)

    def test_transform(self):
        """Test that the method `transform` performs a normalization over categorical values and
        converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = [2, 6]
        values_2 = [[7], [0]]
        values_3 = 0

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)
        result_3 = self.instance.transform(values_3)

        # assert
        expected_result_1 = np.array([[0.25], [0.65]])
        expected_result_2 = np.array([[0.75], [0.05]])
        expected_result_3 = np.array([[0.05]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)
        np.testing.assert_array_equal(result_3, expected_result_3)

    def test_transform_invalid_range(self):
        """Test that ``ValueError`` is being raised when a value out of the hyperparameter space
        is given.
        """
        # run / assert
        with self.assertRaises(ValueError):
            self.instance.transform(10)

        with self.assertRaises(ValueError):
            self.instance.transform([-1, 1])

    def test_inverse_transform(self):
        """Test that the method `inverse_transform` performs a denormalization over the search
        space values to the original categorical hyperparameter space.
        """
        # setup
        values_1 = 0.05
        values_2 = [0.35, 0.65]
        values_3 = np.array([[0.05], [0.35], [0.65]])

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)
        result_3 = self.instance.inverse_transform(values_3)

        # assert
        expected_result_1 = np.array([[0]])
        expected_result_2 = np.array([[3], [6]])
        expected_result_3 = np.array([[0], [3], [6]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)
        np.testing.assert_array_equal(result_3, expected_result_3)

    def test_sample(self):
        """Test that the returned sample values are not bigger than 1 and not smaller than 0."""

        # setup
        n = 10

        # run
        results = self.instance.sample(n)

        # assert
        self.assertEqual(len(results), n)
        assert all(0 < res and res < 1 for res in results)
