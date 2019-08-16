# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from btb.tuning.hyperparams.base import BaseHyperParam, to_array


class TestToArray(TestCase):
    def test_to_array_list_values_valid_dim(self):
        # setup
        values = [1, 2]
        dimension = 1

        # run
        result = to_array(values, dimension)

        # assert
        expected_result = np.array([[1], [2]])

        np.testing.assert_array_equal(result, expected_result)

    def test_to_array_list_values_lt_dimension(self):
        # setup
        values = [1]
        dimension = 2

        # run
        with self.assertRaises(ValueError):
            to_array(values, dimension)

    def test_to_array_list_values_gt_dimension(self):
        # setup
        values = [1, 2, 3]
        dimension = 2

        # run
        with self.assertRaises(ValueError):
            to_array(values, dimension)

    def test_to_array_list_values_eq_dimension(self):
        # setup
        values = [1, 2, 3]
        dimension = 3

        # run
        result = to_array(values, dimension)

        # assert
        expected_result = np.array([[1, 2, 3]])
        np.testing.assert_array_equal(result, expected_result)

    def test_to_array_invalid_values_shape(self):
        # setup
        values = [[[1]]]
        dimension = 1

        # run
        with self.assertRaises(ValueError):
            to_array(values, dimension)

    def test_to_array_invalid_values_dimension(self):
        # setup
        values = [[1, 2], [1, 2, 3]]
        dimension = 2

        # run
        with self.assertRaises(ValueError):
            to_array(values, dimension)


class TestBaseHyperParam(TestCase):

    class TestHyperParam(BaseHyperParam):

        K = 1

        def _inverse_transform(self, values):
            pass

        def _transform(self, values):
            pass

        def sample(self, values):
            pass

    def test__within_range_valid(self):
        # setup
        instance = self.TestHyperParam()
        values = np.array([[1], [2]])
        _min = 0
        _max = 10

        # run / assert
        instance._within_range(values, min=_min, max=_max)

    def test__within_range_invalid(self):
        # setup
        instance = self.TestHyperParam()
        values = np.array([[1], [2]])
        _min = 0
        _max = 1

        # run / assert
        with self.assertRaises(ValueError):
            instance._within_range(values, min=_min, max=_max)

    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_range')
    def test__within_search_space(self, mock__within_range):
        # setup
        instance = self.TestHyperParam()
        values = 15

        # run
        instance._within_search_space(values)

        # assert
        mock__within_range.assert_called_once_with(values, min=0, max=1)

    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_range')
    def test__within_hyperparam_space(self, mock__within_range):
        # setup
        instance = self.TestHyperParam()
        instance.min = 0
        instance.max = 10
        values = 5

        # run
        instance._within_hyperparam_space(values)

        # assert
        mock__within_range.assert_called_once_with(values, min=0, max=10)

    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_search_space')
    @patch('btb.tuning.hyperparams.base.to_array')
    def test_inverse_transform(self, mock_to_array, mock__within_search_space):
        # setup
        instance = self.TestHyperParam()
        instance._inverse_transform = MagicMock()
        values = 1

        # run
        result = instance.inverse_transform(values)

        # assert
        mock_to_array.assert_called_once_with(values, 1)
        mock__within_search_space.assert_called_once_with(mock_to_array.return_value)
        instance._inverse_transform.assert_called_once_with(mock_to_array.return_value)

        self.assertEqual(result, instance._inverse_transform.return_value)

    @patch('btb.tuning.hyperparams.base.np.asarray')
    @patch('btb.tuning.hyperparams.base.BaseHyperParam._transform')
    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_hyperparam_space')
    def test_transform_values_not_ndarray(self, mock__within_hyperparam_space,
                                          mock__transform, mock_np_asarray):
        # setup
        instance = self.TestHyperParam()
        instance._transform = MagicMock()
        values = 1
        mock_np_asarray.return_value = np.array([[1]])

        # run
        result = instance.transform(values)

        # assert
        mock_np_asarray.asser_called_once_with(1)
        mock__within_hyperparam_space.assert_called_once_with(np.array([[1]]))
        instance._transform.assert_called_once_with(np.array([[1]]))
        self.assertEqual(result, instance._transform.return_value)

    @patch('btb.tuning.hyperparams.base.BaseHyperParam._transform')
    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_hyperparam_space')
    def test_transform_values_list(self, mock__within_hyperparam_space, mock__transform):
        # setup
        instance = self.TestHyperParam()
        instance._transform = MagicMock()
        values = np.array([[1]])

        # run
        result = instance.transform(values)

        # assert
        mock__within_hyperparam_space.assert_called_once_with(values)
        instance._transform.assert_called_once_with(values)
        self.assertEqual(result, instance._transform.return_value)
