# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from btb.tuning.acquisition.expected_improvement import ExpectedImprovementAcquisition


class TestExpectedImprovementAcquisition(TestCase):

    @patch('btb.tuning.acquisition.expected_improvement.super')
    @patch('btb.tuning.acquisition.expected_improvement.norm')
    def test__acquire(self, mock_norm, mock_super):
        # setup
        instance = MagicMock()
        instance._scores = np.array([1, 2])
        instance.maximize = True
        candidates = np.array([3, 4])
        mock_super.return_value._acquire.return_value = np.array([0])
        mock_norm.cdf.return_value = 1
        mock_norm.pdf.return_value = 1

        # assert
        result = ExpectedImprovementAcquisition._acquire(instance, candidates)

        # assert
        mock_norm.cdf.assert_called_once_with(0.25)
        mock_norm.pdf.assert_called_once_with(0.25)
        mock_super.return_value._acquire.assert_called_once_with(np.array([5]), 1)

        np.testing.assert_array_equal(result, np.array([0]))
