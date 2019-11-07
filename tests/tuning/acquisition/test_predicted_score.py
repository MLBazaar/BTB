# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, call

import numpy as np

from btb.tuning.acquisition.predicted_score import PredictedScoreAcquisition


def assert_called_with_np_array(mock_calls, real_calls):
    assert len(mock_calls) == len(real_calls)

    for mock_call, real_call in zip(mock_calls, real_calls):
        np.testing.assert_array_equal(mock_call[0][0], real_call[1][0])
        assert mock_call[0][1] == real_call[1][1]


class TestPredictedScoreAcquisition(TestCase):

    def test__acquire(self):
        # setup
        candidates = np.array([[1, 2]])
        instance = MagicMock()
        instance._get_max_candidates.return_value = 'max_candidate'

        # assert
        result = PredictedScoreAcquisition._acquire(instance, candidates)

        # assert
        called_with = instance._get_max_candidates.call_args_list
        expected_call = [call(np.array([1]), 1)]
        assert_called_with_np_array(called_with, expected_call)
        assert result == 'max_candidate'

    def test__acquire_num_candidates_gt_one(self):
        # setup
        candidates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        instance = MagicMock()
        instance._get_max_candidates.return_value = 'max_candidates'

        # assert
        result = PredictedScoreAcquisition._acquire(instance, candidates, num_candidates=2)

        # assert
        called_with = instance._get_max_candidates.call_args_list
        expected_call = [call(np.array([1, 2, 3, 4]), 2)]
        assert_called_with_np_array(called_with, expected_call)
        assert result == 'max_candidates'

    def test__acquire_candidates_shape_one(self):
        # setup
        candidates = np.array([1, 2, 3])
        instance = MagicMock()
        instance._get_max_candidates.return_value = 'max_candidate'

        # assert
        result = PredictedScoreAcquisition._acquire(instance, candidates, num_candidates=1)

        # assert
        called_with = instance._get_max_candidates.call_args_list
        expected_call = [call(np.array([1, 2, 3]), 1)]
        assert_called_with_np_array(called_with, expected_call)
        assert result == 'max_candidate'
