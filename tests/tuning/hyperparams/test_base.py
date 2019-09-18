# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from btb.tuning.hyperparams.base import BaseHyperParam


class TestBaseHyperParam(TestCase):

    class TestHyperParam(BaseHyperParam):

        K = 1

        def _inverse_transform(self, values):
            pass

        def _transform(self, values):
            pass

        def sample(self, values):
            pass

    def setUp(self):
        self.instance = self.TestHyperParam()

    def test__within_range_valid(self):
        # setup
        values = np.array([[1], [2]])
        _min = 0
        _max = 10

        # run / assert
        self.instance._within_range(values, min=_min, max=_max)

    def test__within_range_invalid(self):
        # setup
        values = np.array([[1], [2]])
        _min = 0
        _max = 1

        # run / assert
        with self.assertRaises(ValueError):
            self.instance._within_range(values, min=_min, max=_max)

    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_range')
    def test__within_search_space(self, mock__within_range):
        # setup
        values = 15

        # run
        self.instance._within_search_space(values)

        # assert
        mock__within_range.assert_called_once_with(15, min=0, max=1)

    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_range')
    def test__within_hyperparam_space(self, mock__within_range):
        # setup
        self.instance.min = 0
        self.instance.max = 10
        values = 5

        # run
        self.instance._within_hyperparam_space(values)

        # assert
        mock__within_range.assert_called_once_with(5, min=0, max=10)

    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_search_space')
    @patch('btb.tuning.hyperparams.base.BaseHyperParam.to_array')
    def test_inverse_transform(self, mock_to_array, mock__within_search_space):
        # setup
        self.instance._inverse_transform = MagicMock(return_value=3)
        mock_to_array.return_value = 2
        values = 1

        # run
        result = self.instance.inverse_transform(values)

        # assert
        mock_to_array.assert_called_once_with(1)
        mock__within_search_space.assert_called_once_with(2)
        self.instance._inverse_transform.assert_called_once_with(2)

        self.assertEqual(result, 3)

    @patch('btb.tuning.hyperparams.base.np.asarray')
    @patch('btb.tuning.hyperparams.base.BaseHyperParam._transform')
    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_hyperparam_space')
    def test_transform_values_not_ndarray(self, mock__within_hyperparam_space,
                                          mock__transform, mock_np_asarray):
        # setup
        self.instance._transform = MagicMock(return_value=2)
        values = 1
        mock_np_asarray.return_value = np.array([[1]])

        # run
        result = self.instance.transform(values)

        # assert
        mock_np_asarray.asser_called_once_with(1)
        mock__within_hyperparam_space.assert_called_once_with(np.array([[1]]))
        self.instance._transform.assert_called_once_with(np.array([[1]]))
        self.assertEqual(result, 2)

    @patch('btb.tuning.hyperparams.base.BaseHyperParam._transform')
    @patch('btb.tuning.hyperparams.base.BaseHyperParam._within_hyperparam_space')
    def test_transform_values_list(self, mock__within_hyperparam_space, mock__transform):
        # setup
        self.instance = self.TestHyperParam()
        self.instance._transform = MagicMock(return_value=2)
        values = np.array([[1]])

        # run
        result = self.instance.transform(values)

        # assert
        mock__within_hyperparam_space.assert_called_once_with(np.array([[1]]))
        self.instance._transform.assert_called_once_with(np.array([[1]]))
        self.assertEqual(result, 2)

    def test_to_array_list_values_valid_dim(self):
        # setup
        values = [1, 2]

        # run
        result = self.instance.to_array(values)

        # assert
        expected_result = np.array([[1], [2]])

        np.testing.assert_array_equal(result, expected_result)

    def test_to_array_list_values_lt_dimension(self):
        # setup
        values = [1]
        self.instance.K = 2

        # run
        with self.assertRaises(ValueError):
            self.instance.to_array(values)

    def test_to_array_list_values_gt_dimension(self):
        # setup
        values = [1, 2, 3]
        self.instance.K = 2

        # run
        with self.assertRaises(ValueError):
            self.instance.to_array(values)

    def test_to_array_list_values_eq_dimension(self):
        # setup
        values = [1, 2, 3]
        self.instance.K = 3

        # run
        result = self.instance.to_array(values)

        # assert
        expected_result = np.array([[1, 2, 3]])
        np.testing.assert_array_equal(result, expected_result)

    def test_to_array_invalid_values_shape(self):
        # setup
        values = [[[1]]]
        self.instance.K = 1

        # run
        with self.assertRaises(ValueError):
            self.instance.to_array(values)

    def test_to_array_invalid_values_dimension(self):
        # setup
        values = [[1, 2], [1, 2, 3]]
        self.instance.K = 2

        # run
        with self.assertRaises(ValueError):
            self.instance.to_array(values)
