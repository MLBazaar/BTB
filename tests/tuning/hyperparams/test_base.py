# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from btb.tuning.hyperparams.base import BaseHyperParam


class TestBaseHyperParam(TestCase):

    def test__within_range_valid_range(self):
        # setup
        instance = MagicMock()
        values = np.array([[1], [2]])
        _min = 0
        _max = 10

        # run / assert
        BaseHyperParam._within_range(instance, values, min=_min, max=_max)

    def test__within_range_invalid_range(self):
        # setup
        instance = MagicMock()
        values = np.array([[1], [2]])
        _min = 0
        _max = 1

        # run / assert
        with self.assertRaises(ValueError):
            BaseHyperParam._within_range(instance, values, min=_min, max=_max)

    def test__within_search_space(self):
        # setup
        instance = MagicMock()
        values = np.array(15)

        # run
        BaseHyperParam._within_search_space(instance, values)

        # assert
        instance._within_range.assert_called_once_with(np.array(15), min=0, max=1)

    def test__within_hyperparam_space(self):
        # setup
        instance = MagicMock()
        instance.min = 0
        instance.max = 10
        values = 5

        # run
        BaseHyperParam._within_hyperparam_space(instance, values)

        # assert
        instance._within_range.assert_called_once_with(5, min=0, max=10)

    def test_inverse_transform(self):
        # setup
        instance = MagicMock()
        instance._inverse_transform = MagicMock(return_value=3)
        instance._to_array.return_value = 2
        values = 1

        # run
        result = BaseHyperParam.inverse_transform(instance, values)

        # assert
        instance._to_array.assert_called_once_with(1)
        instance._within_search_space.assert_called_once_with(2)
        instance._inverse_transform.assert_called_once_with(2)

        assert result == 3

    @patch('btb.tuning.hyperparams.base.np.asarray')
    def test_transform_values_not_ndarray(self, mock_np_asarray):
        # setup
        instance = MagicMock()
        instance._transform.return_value = 2
        mock_np_asarray.return_value = np.array([[1]])
        values = 1

        # run
        result = BaseHyperParam.transform(instance, values)

        # assert
        mock_np_asarray.asser_called_once_with(1)
        instance._within_hyperparam_space.assert_called_once_with(np.array([[1]]))
        instance._transform.assert_called_once_with(np.array([[1]]))
        self.assertEqual(result, 2)

    def test_transform_values_list(self):
        # setup
        instance = MagicMock()
        instance._transform.return_value = 2
        values = np.array([[1]])

        # run
        result = BaseHyperParam.transform(instance, values)

        # assert
        instance._within_hyperparam_space.assert_called_once_with(np.array([[1]]))
        instance._transform.assert_called_once_with(np.array([[1]]))
        assert result == 2

    @patch('btb.tuning.hyperparams.base.np.asarray')
    def test_transform_dimensions_gt_two(self, mock_asarray):
        # setup
        array = MagicMock()
        array.shape.__len__.return_value = 3

        mock_asarray.return_value = array

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam.transform(MagicMock(), 1)

    def test__to_array_scalar_value_dimension_one(self):
        # setup
        instance = MagicMock()
        instance.dimensions = 1
        values = 1

        # run
        result = BaseHyperParam._to_array(instance, values)

        # assert
        np.testing.assert_array_equal(result, np.array([[1]]))

    def test__to_array_scalar_value_dimensions_gt_one(self):
        # setup
        instance = MagicMock()
        instance.dimensions = 2

        values = 1

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)

    def test__to_array_list_values_of_scalar_values_dimensions_one(self):
        # setup
        instance = MagicMock()
        instance.dimensions = 1
        values = [1, 2, 3]

        # run
        result = BaseHyperParam._to_array(instance, values)

        # assert
        np.testing.assert_array_equal(result, np.array([[1], [2], [3]]))

    def test__to_array_list_values_of_scalar_values_dimensions_gt_one(self):
        # setup
        instance = MagicMock()
        instance.dimensions = 2
        values = [1, 2, 3]

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)

    def test__to_array_list_values_of_list_values_dimensions_one(self):
        # setup
        instance = MagicMock()
        instance.dimensions = 1
        values = [[1], [2], [3]]

        # run
        result = BaseHyperParam._to_array(instance, values)

        # assert
        np.testing.assert_array_equal(result, np.array([[1], [2], [3]]))

    def test__to_array_list_values_of_list_values_dimensions_two(self):

        # setup
        instance = MagicMock()
        instance.dimensions = 2
        values = [[1], [2], [3]]

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)

    def test__to_array_list_values_of_list_values_one_scalar(self):
        # setup
        instance = MagicMock()
        instance.dimensions = 1
        values = [[1], 2, [3]]

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)

    def test__to_array_values_shape_gt_two(self):
        # setup
        instance = MagicMock()
        instance.dimensions = 1
        values = [[[1], [3]]]

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)

    @patch('btb.tuning.hyperparams.base.np.array')
    def test__to_array_len_shape_is_one(self, mock_np_array):
        # setup
        instance = MagicMock()
        instance.dimensions = 1
        values = [1]
        array = MagicMock()
        array.shape.__len__.return_value = 1

        mock_np_array.return_value = array

        # run
        BaseHyperParam._to_array(instance, values)

        # assert
        array.reshape.assert_called_once_with(-1, 1)

    @patch('btb.tuning.hyperparams.base.np.array')
    def test__to_array_more_than_one_column_for_dimensions_one(self, mock_np_array):
        # setup
        instance = MagicMock()
        instance.dimensions = 1
        values = [1]
        array = MagicMock()
        array.shape.__len__.return_value = 2
        array.shape = [2, 3]

        mock_np_array.return_value = array

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)

    @patch('btb.tuning.hyperparams.base.np.array')
    def test__to_array_values_shape_one_dimensions_two(self, mock_np_array):

        # setup
        instance = MagicMock()
        instance.dimensions = 2
        values = [1]
        array = MagicMock()
        array.__len__.return_value = 3
        array.shape.__len__.return_value = 1

        mock_np_array.return_value = array

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)

    @patch('btb.tuning.hyperparams.base.np.isscalar')
    @patch('btb.tuning.hyperparams.base.np.array')
    def test__to_array_values_not_scalar_dimensions_two(self, mock_np_array, mock_np_isscalar):

        # setup
        instance = MagicMock()
        instance.dimensions = 2
        mock_np_isscalar.return_value = False
        values = [1]
        array = MagicMock()
        array.__len__.return_value = 2
        array.shape.__len__.return_value = 1

        mock_np_array.return_value = array

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)

    @patch('btb.tuning.hyperparams.base.np.isscalar')
    @patch('btb.tuning.hyperparams.base.np.array')
    def test__to_array_values_reshape_dimensions_two(self, mock_np_array, mock_np_isscalar):

        # setup
        mock_np_isscalar.side_effect = [False, True]

        instance = MagicMock()
        instance.dimensions = 2
        mock_np_isscalar.return_value = False
        values = [1]
        array = MagicMock()
        array.__len__.return_value = 2
        array.shape.__len__.return_value = 1

        mock_np_array.return_value = array

        # run
        BaseHyperParam._to_array(instance, values)
        array.reshape.assert_called_once_with(1, -1)

    @patch('btb.tuning.hyperparams.base.np.array')
    def test__to_array_len_shape_is_gt_two(self, mock_np_array):
        # setup
        instance = MagicMock()
        instance.dimensions = 1
        values = [1]
        array = MagicMock()
        array.shape.__len__.return_value = 3

        mock_np_array.return_value = array

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)

    @patch('btb.tuning.hyperparams.base.np.isscalar')
    @patch('btb.tuning.hyperparams.base.np.array')
    def test__to_array_not_all_scalars(self, mock_np_array, mock_np_isscalar):
        # setup
        instance = MagicMock()
        instance.dimensions = 2
        values = [1, 2]
        array = MagicMock()
        array.shape.__len__.return_value = 1
        array.__len__.return_value = 2
        array.__iter__.return_value = [1]

        mock_np_isscalar.side_effect = [False, True, True, False, False]
        mock_np_array.return_value = array

        # run
        with self.assertRaises(ValueError):
            BaseHyperParam._to_array(instance, values)
