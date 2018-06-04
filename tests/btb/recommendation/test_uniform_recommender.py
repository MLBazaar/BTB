from unittest import TestCase

import numpy as np
from mock import patch

from btb.recommendation.uniform import UniformRecommender


class TestBaseRecommender(TestCase):
    def setUp(self):
        # Set-up
        self.dpp_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0],
            [0, 4, 0, 6, 0, 4, 0, 2, 1, 0, 0, 2, 3, 1, 0, 0],
            [1, 0, 1, 2, 0, 0, 6, 1, 0, 5, 1, 0, 0, 0, 0, 1],
            [0, 2, 3, 0, 0, 0, 0, 0, 4, 1, 3, 2, 0, 0, 1, 4]
        ])
        self.dpp_matrix_small = np.array([
            [1, 0, 0, 0],
            [0, 4, 0, 6],
            [1, 0, 1, 2],
            [0, 2, 3, 0]
        ])

    @patch('btb.recommendation.uniform.np.random.permutation')
    def test_predict_all(self, randpermutation_mock):
        indicies = np.array(range(4))
        permutation = [3, 2, 1, 4]
        randpermutation_mock.return_value = permutation
        recommender = UniformRecommender(self.dpp_matrix_small)
        predictions = recommender.predict(indicies)
        np.testing.assert_array_equal(
            permutation,
            predictions,
        )

    @patch('btb.recommendation.uniform.np.random.permutation')
    def test_predict_one(self, randpermutation_mock):
        indicies = np.array([0])
        permutation = [1]
        randpermutation_mock.return_value = permutation
        recommender = UniformRecommender(self.dpp_matrix)
        predictions = recommender.predict(indicies)
        np.testing.assert_array_equal(
            permutation,
            predictions,
        )
