# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from btb.tuning.acquisition.expected_improvement import ExpectedImprovementAcquisition


class TestExpectedImprovementAcquisition(TestCase):

    def test__acquire(self):
        # run
        instance = ExpectedImprovementAcquisition()
        instance.scores = np.array([0.5, 0.6, 0.7])

        predictions = np.array([
            [0.8, 1],
            [0.9, 2]
        ])
        best = instance._acquire(predictions)

        # assert
        np.testing.assert_array_equal(best, np.array([1]))

    def test__acquire_n_candidates(self):
        # run
        instance = ExpectedImprovementAcquisition()
        instance.scores = np.array([0.5, 0.9, 0.7, 0.8])

        predictions = np.array([
            [0.1, 1],
            [0.9, 9],
            [0.7, 3],
            [0.75, 4]
        ])
        best = instance._acquire(predictions, 2)

        # assert
        np.testing.assert_array_equal(best, np.array([1, 3]))
