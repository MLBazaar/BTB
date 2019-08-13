# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from btb.tuning.hyperparams.boolean import BooleanHyperParam


class TestBooleanHyperParam(TestCase):
    """Unit test for the class ``BooleanHyperParam``."""

    @classmethod
    def setUpClass(cls):
        cls.instance = BooleanHyperParam()

    def test_transform_list_values(self):
        """Test that the method ``transform`` performs a normalization over a ``list`` of boolean
        values and converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = [True, False, True]
        values_2 = np.asarray([[True], [False]])

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)

        # assert
        expected_result_1 = np.asarray([[1], [0], [1]])
        expected_result_2 = np.asarray([[1], [0]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)

    def test_transform_list_out_of_shape(self):
        """Test that a ``ValueError`` is being raised when an invalid shape of values has been
        given.
        """
        # setup
        invalid_values = [[True, False]]

        # run / assert
        with self.assertRaises(ValueError):
            self.instance.transform(invalid_values)

    def test_transform_sinlge_value(self):
        """Test that the method ``transform`` performs a normalization over a sinlge boolean
        value and converts them in to a search space of [0, 1]^k.
        """

        # setup
        values_1 = False
        values_2 = [True]

        # run
        result_1 = self.instance.transform(values_1)
        result_2 = self.instance.transform(values_2)

        # assert
        expected_result_1 = np.asarray([[0]])
        expected_result_2 = np.asarray([[1]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)

    def test_transform_invalid_values(self):
        """Test that ``ValueError`` is being raised when a value out of the hyperparameter space
        is given.
        """

        # run / assert
        with self.assertRaises(ValueError):
            self.instance.transform(1)

    def test_inverse_transform_list_values(self):
        """Test that the method ``inverse_transform`` performs a denormalization over the search
        space values from a list to the original hyperparameter space.
        """

        # setup
        values_1 = [0, 1, 0]
        values_2 = [[1], [0], [1]]

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)

        # assert
        expected_result_1 = np.asarray([[False], [True], [False]])
        expected_result_2 = np.asarray([[True], [False], [True]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)

    def test_inverse_transform_single_value(self):
        """Test that the method ``inverse_transform`` performs a denormalization over the search
        space values from a sinlge to the original hyperparameter space.
        """
        # setup
        values_1 = 0
        values_2 = [1]

        # run
        result_1 = self.instance.inverse_transform(values_1)
        result_2 = self.instance.inverse_transform(values_2)

        # assert
        expected_result_1 = np.asarray([[False]])
        expected_result_2 = np.asarray([[True]])

        np.testing.assert_array_equal(result_1, expected_result_1)
        np.testing.assert_array_equal(result_2, expected_result_2)

    def test_sample(self):
        """Test that the method ``sample`` returns values from the search space and not the
        original hyperparameter space.
        """

        # setup
        n = 10

        # run
        result = self.instance.sample(n)

        # assert
        self.assertEqual(len(result), n)
        assert all(0 == res or res == 1 for res in result)
