# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from btb.tuning.acquisition.numpyargsort import NumpyArgSortFunction


class TestNumpyArgSortFunction(TestCase):

    def test__acquire(self):
        # setup
        instance = MagicMock()
        candidates = np.array([[1, 2]])

        # assert
        result = NumpyArgSortFunction._acquire(instance, candidates)

        # assert
        np.testing.assert_array_equal(result, np.array([0]))

    def test__acquire_num_candidates_gt_one(self):
        # setup
        instance = MagicMock()
        candidates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

        # assert
        result = NumpyArgSortFunction._acquire(instance, candidates, num_candidates=2)

        # assert
        np.testing.assert_array_equal(result, np.array([3, 2]))

    def test__acquire_candidates_shape_one(self):
        # setup
        instance = MagicMock()
        candidates = np.array([1, 2, 3])

        # assert
        result = NumpyArgSortFunction._acquire(instance, candidates, num_candidates=1)

        # assert
        np.testing.assert_array_equal(result, np.array([2]))