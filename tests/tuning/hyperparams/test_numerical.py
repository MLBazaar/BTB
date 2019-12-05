# -*- coding: utf-8 -*-

import sys
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from btb.tuning.hyperparams.numerical import FloatHyperParam, IntHyperParam


class TestFloatHyperParam(TestCase):

    def test___init__no_min_no_max(self):
        """Test instantiation with ``min=None`` and ``max=None``"""
        # run
        instance = FloatHyperParam()

        # assert
        self.assertEqual(instance.min, sys.float_info.min)
        self.assertEqual(instance.max, sys.float_info.max)
        self.assertEqual(instance.default, sys.float_info.min)

    def test___init__with_np_inf(self):
        """Test instantiation with ``min=-np.inf`` and ``max=np.inf``"""
        # run
        instance = FloatHyperParam(min=-np.inf, max=np.inf)

        # assert
        self.assertEqual(instance.min, sys.float_info.min)
        self.assertEqual(instance.max, sys.float_info.max)
        self.assertEqual(instance.default, sys.float_info.min)

    def test___init__min_no_max(self):
        """Test instantiation with ``min=n`` and ``max=None``"""
        # setup
        _min = 0.1

        # run
        instance = FloatHyperParam(min=_min, max=None)

        # assert
        self.assertEqual(instance.min, _min)
        self.assertEqual(instance.max, sys.float_info.max)
        self.assertEqual(instance.default, 0.1)

    def test___init__no_min_max(self):
        """Test instantiation with ``min=None`` and ``max=n``"""
        # setup
        _max = 0.1

        # run
        instance = FloatHyperParam(min=None, max=_max)

        # assert
        self.assertEqual(instance.min, sys.float_info.min)
        self.assertEqual(instance.max, _max)
        self.assertEqual(instance.default, sys.float_info.min)

    def test___init__min_eq_max(self):
        """Test instantiation with ``min=n`` and ``max=n``"""
        # setup
        n = 0.1

        # run / assert
        with self.assertRaises(ValueError):
            FloatHyperParam(min=n, max=n)

    def test___init__min_gt_max(self):
        """Test instantiation with ``min`` being reater than ``max``"""

        # run / assert
        with self.assertRaises(ValueError):
            FloatHyperParam(min=1, max=0)

    def test___init__min_max(self):
        """Test instantiation with ``min=n`` and ``max=x``"""
        # setup
        _min = 0.1
        _max = 0.9

        # run
        instance = FloatHyperParam(min=_min, max=_max, default=1)

        # assert
        self.assertEqual(instance.min, _min)
        self.assertEqual(instance.max, _max)
        self.assertEqual(instance.default, 1.0)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__transform_no_min_no_max(self, mock_sys):
        """Test that the method ``_transform`` performs a normalization of values between ``min``
        and ``max`` with no limit set on them.
        """
        # setup
        mock_sys.float_info.max = 1000.0  # This values can be different in each OS.
        mock_sys.float_info.min = 0.0

        instance = FloatHyperParam()
        values = np.array([[0.9], [100.8]])

        # run
        result = instance._transform(values)

        # assert
        expected_result = np.array([[0.0009],
                                    [0.1008]])

        np.testing.assert_array_equal(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__transform_min_no_max(self, mock_sys):
        """Test that the method ``_transform`` performs a normalization of values between ``min``
        and ``max`` with a min value set in.
        """
        # setup
        mock_sys.float_info.max = 10.0
        _min = 1.0

        instance = FloatHyperParam(min=_min)
        values = np.array([[7.3], [4.6]])

        # run
        result = instance._transform(values)

        # assert
        expected_result = np.array([[0.7],
                                    [0.4]])

        np.testing.assert_allclose(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__transform_no_min_max(self, mock_sys):
        """Test that the method ``_transform`` performs a normalization of values between ``min``
        and ``max`` with a max value set in.
        """
        # setup
        mock_sys.float_info.min = 0.0
        _max = 1.0

        instance = FloatHyperParam(max=_max)
        values = np.array([[0.1], [0.9]])

        # run
        result = instance._transform(values)

        # assert
        expected_result = np.array([[0.1],
                                    [0.9]])

        np.testing.assert_array_equal(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__transform_min_max(self, mock_sys):
        """Test that the method ``_transform`` performs a normalization of values between ``min``
        and ``max`` with a max and min value set in.
        """
        # setup
        _min = 0.0
        _max = 10.0
        instance = FloatHyperParam(min=_min, max=_max)
        values = np.array([[0.1], [0.9]])

        # run
        result = instance._transform(values)

        # assert
        expected_result = np.array([[0.01],
                                    [0.09]])

        np.testing.assert_array_equal(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__inverse_transform_no_min_no_max(self, mock_sys):
        """Test that the method ``_inverse_transform`` performs a normalization of values between
        ``min`` and ``max`` with no min or max set.
        """
        # setup
        mock_sys.float_info.max = 1000.0
        mock_sys.float_info.min = 0.0

        instance = FloatHyperParam()
        values = np.array([[0.1], [0.2]])

        # run
        result = instance._inverse_transform(values)

        # assert
        expected_result = np.array([[100.],
                                    [200.]])

        np.testing.assert_array_equal(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__inverse_transform_min_no_max(self, mock_sys):
        """Test that the method ``_inverse_transform`` performs a normalization of values between
        ``min`` and ``max`` with min value set up.
        """
        # setup
        mock_sys.float_info.max = 10.0
        _min = 1.0

        instance = FloatHyperParam(min=_min)
        values = np.array([[0.1], [0.]])

        # run
        result = instance._inverse_transform(values)

        # assert
        expected_result = np.array([[1.9],
                                    [1.0]])

        np.testing.assert_array_equal(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__inverse_transform_no_min_max(self, mock_sys):
        """Test that the method ``_inverse_transform`` performs a normalization of values between
        ``min`` and ``max`` with a max value set up.
        """
        # setup
        mock_sys.float_info.min = 0.0
        _max = 1.0

        instance = FloatHyperParam(max=_max)
        values = np.array([[0.1], [0.9]])

        # run
        result = instance._inverse_transform(values)

        # assert
        expected_result = np.array([[0.1],
                                    [0.9]])

        np.testing.assert_array_equal(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__inverse_transform_min_max(self, mock_sys):
        """Test that the method ``_inverse_transform`` performs a normalization of values between
        ``min`` and ``max`` with a min and max value set up.
        """
        # setup
        _min = 0.0
        _max = 10.0

        instance = FloatHyperParam(min=_min, max=_max)
        values = np.array([[0.1], [0.9]])

        # run
        result = instance._inverse_transform(values)

        # assert
        expected_result = np.array([[1.],
                                    [9.]])

        np.testing.assert_array_equal(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.np.random.random')
    def test_sample(self, mock_np_random):
        """Test that the method ``sample`` is being called with `n_samples and
        `self.dimensions`.
        """
        # setup
        mock_np_random.return_value = np.array([[0.1], [0.2]])
        instance = FloatHyperParam()
        n_samples = 2

        # run
        result = instance.sample(n_samples)

        # assert
        expected_result = np.array([[0.1], [0.2]])

        mock_np_random.assert_called_once_with((n_samples, instance.dimensions))
        np.testing.assert_array_equal(result, expected_result)


class TestIntHyperParam(TestCase):

    def test___init__no_min_no_max(self):
        """Test instantiation with ``min=None`` and ``max=None``"""
        # run
        instance = IntHyperParam()

        # assert
        expected_min = -(sys.maxsize / 2)
        expected_max = sys.maxsize / 2

        self.assertEqual(instance.min, expected_min)
        self.assertEqual(instance.max, expected_max)
        self.assertEqual(instance.default, expected_min)
        self.assertEqual(instance.step, 1)

    def test___init__exclude_min_no_max(self):
        """Test instantiation with ``min=None`` and ``max=None`` excluding ``min``."""
        # run
        instance = IntHyperParam(include_min=False)

        # assert
        expected_min = int(-(sys.maxsize / 2)) + 1
        expected_max = int(sys.maxsize / 2)

        self.assertEqual(instance.min, expected_min)
        self.assertEqual(instance.max, expected_max)
        self.assertEqual(instance.step, 1)

    def test___init__no_min_exclude_max(self):
        """Test instantiation with ``min=None`` and ``max=None`` excluding ``max``."""
        # run
        instance = IntHyperParam(include_max=False)

        # assert
        expected_min = int(-(sys.maxsize / 2))
        expected_max = int(sys.maxsize / 2) - 1

        self.assertEqual(instance.min, expected_min)
        self.assertEqual(instance.max, expected_max)
        self.assertEqual(instance.default, expected_min)
        self.assertEqual(instance.step, 1)

    def test___init__min_no_max(self):
        """Test instantiation with ``min=n`` and ``max=None``"""
        # setup
        _min = 1

        # run
        instance = IntHyperParam(min=_min, max=None)

        # assert
        self.assertEqual(instance.min, _min)
        self.assertEqual(instance.max, sys.maxsize / 2)
        self.assertEqual(instance.default, 1)
        self.assertEqual(instance.step, 1)

    def test___init__no_min_max(self):
        """Test instantiation with ``min=None`` and ``max=n``"""
        # setup
        _max = 1

        # run
        instance = IntHyperParam(min=None, max=_max)

        # assert
        expected_min = -(sys.maxsize / 2)
        expected_max = 1
        self.assertEqual(instance.min, expected_min)
        self.assertEqual(instance.max, expected_max)
        self.assertEqual(instance.step, 1)

    def test___init__min_eq_max(self):
        """Test instantiation with ``min=n`` and ``max=n``"""
        # setup
        n = 1

        # run / assert
        with self.assertRaises(ValueError):
            IntHyperParam(min=n, max=n)

    def test___init__min_gt_max(self):
        """Test instantiation with ``min`` being reater than ``max``"""

        # run / assert
        with self.assertRaises(ValueError):
            IntHyperParam(min=1, max=0)

    def test___init__min_max(self):
        """Test instantiation with ``min=n`` and ``max=x``"""
        # setup
        _min = 1
        _max = 9

        # run
        instance = IntHyperParam(min=_min, max=_max, default=5)

        # assert
        self.assertEqual(instance.min, _min)
        self.assertEqual(instance.max, _max)
        self.assertEqual(instance.default, 5)
        self.assertEqual(instance.step, 1)

    def test___init__min_max_step(self):
        """Test instantiation with ``min=n`` and ``max=x`` and step."""
        # setup
        _min = 0
        _max = 10
        _step = 2

        # run
        instance = IntHyperParam(min=_min, max=_max, step=_step)

        # assert
        self.assertEqual(instance.min, 0)
        self.assertEqual(instance.max, 10)
        self.assertEqual(instance.step, 2)

    def test___init__min_max_invalid_step(self):
        """Test instantiation with ``min=n`` and ``max=x``"""
        # setup
        _min = 1
        _max = 9
        _step = 5

        # run / assert
        with self.assertRaises(ValueError):
            IntHyperParam(min=_min, max=_max, step=_step)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__transform_no_min_no_max(self, mock_sys):
        """Test that the method ``_transform`` performs a normalization of values between ``min``
        and ``max`` with no limit set on them.
        """
        # setup
        mock_sys.maxsize = 1000  # This values can be different in each OS.
        instance = IntHyperParam()
        values = np.array([[9], [100]])

        # run
        result = instance._transform(values)

        # assert
        expected_result = np.array([[0.50899101],
                                    [0.5999001]])

        np.testing.assert_allclose(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__transform_min_no_max(self, mock_sys):
        """Test that the method ``_transform`` performs a normalization of values between ``min``
        and ``max`` with a min value set in.
        """
        # setup
        mock_sys.maxsize = 1000
        _min = 1
        instance = IntHyperParam(min=_min)
        values = np.array([[5], [1]])

        # run
        result = instance._transform(values)

        # assert
        expected_result = np.array([[0.009],
                                    [0.001]])

        np.testing.assert_allclose(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__transform_no_min_max(self, mock_sys):
        """Test that the method ``_transform`` performs a normalization of values between ``min``
        and ``max`` with a max value set in.
        """
        # setup
        mock_sys.maxsize = 1000
        _max = 10
        instance = IntHyperParam(max=_max)
        values = np.array([[1], [9]])

        # run
        result = instance._transform(values)

        # assert
        expected_result = np.array([[0.981409],
                                    [0.99706458]])

        np.testing.assert_allclose(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__transform_min_max(self, mock_sys):
        """Test that the method ``_transform`` performs a normalization of values between ``min``
        and ``max`` with a max and min value set in.
        """
        # setup
        _min = 0
        _max = 10

        instance = IntHyperParam(min=_min, max=_max)
        values = np.array([[9], [1]])

        # run
        result = instance._transform(values)

        # assert
        expected_result = np.array([[0.86363636],
                                    [0.13636364]])

        np.testing.assert_allclose(result, expected_result)

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__inverse_transform_no_min_no_max(self, mock_sys):
        """Test that the method ``_inverse_transform`` performs a normalization of values between
        ``min`` and ``max`` with no min or max set.
        """
        # setup
        mock_sys.maxsize = 1000
        instance = IntHyperParam()
        values = np.array([[0.0009], [0.1008]])

        # run
        result = instance._inverse_transform(values)

        # assert
        expected_result = np.array([[-500],
                                    [-400]])

        np.testing.assert_array_equal(result, expected_result.astype(int))

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__inverse_transform_min_no_max(self, mock_sys):
        """Test that the method ``_inverse_transform`` performs a normalization of values between
        ``min`` and ``max`` with min value set up.
        """
        # setup
        mock_sys.maxsize = 1000
        _min = 1
        instance = IntHyperParam(min=_min)
        values = np.array([[0.1], [0.]])

        # run
        result = instance._inverse_transform(values)

        # assert
        expected_result = np.array([[50],
                                    [1]])

        np.testing.assert_array_equal(result, expected_result.astype(int))

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__inverse_transform_no_min_max(self, mock_sys):
        """Test that the method ``_inverse_transform`` performs a normalization of values between
        ``min`` and ``max`` with a max value set up.
        """
        # setup
        mock_sys.maxsize = 1000
        _max = 500
        instance = IntHyperParam(max=_max)
        values = np.array([[0.1], [0.9]])

        # run
        result = instance._inverse_transform(values)

        # assert
        expected_result = np.array([[-400],
                                    [400]])

        np.testing.assert_array_equal(result, expected_result.astype(int))

    @patch('btb.tuning.hyperparams.numerical.sys')
    def test__inverse_transform_min_max(self, mock_sys):
        """Test that the method ``_inverse_transform`` performs a normalization of values between
        ``min`` and ``max`` with a min and max value set up.
        """
        # setup
        _min = 0
        _max = 10
        instance = IntHyperParam(min=_min, max=_max)
        values = np.array([[0.1], [0.9]])

        # run
        result = instance._inverse_transform(values)

        # assert
        expected_result = np.array([[1],
                                    [9]])

        np.testing.assert_array_equal(result, expected_result.astype(int))

    @patch('btb.tuning.hyperparams.numerical.np.random.random')
    @patch('btb.tuning.hyperparams.numerical.IntHyperParam._transform')
    @patch('btb.tuning.hyperparams.numerical.IntHyperParam._inverse_transform')
    def test_sample(self, mock__inverse_transform, mock__transform, mock_np_random):
        """Test that the method ``sample`` returns random generated numbers and process them thro
        the internal methods to convert them in the range of our search space."""
        # setup
        instance = IntHyperParam()
        n_samples = 2

        # run
        result = instance.sample(n_samples)

        # assert
        mock_np_random.assert_called_once_with((n_samples, instance.dimensions))
        mock__inverse_transform.assert_called_once_with(mock_np_random.return_value)
        mock__transform.assert_called_once_with(mock__inverse_transform.return_value)
        self.assertEqual(result, mock__transform.return_value)
