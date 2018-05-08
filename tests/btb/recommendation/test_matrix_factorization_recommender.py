from unittest import TestCase

import numpy as np
from mock import patch

from btb.recommendation.matrix_factorization import MFRecommender


class TestBaseRecommender(TestCase):
    def setUp(self):
        self.n_components = 3
        self.r_min = 0
        self.dpp_decomposed = np.array([
            [.0, .2, .4],
            [.8, .6, .4],
            [1.0, .8, 0.9],
            [.6, 1.0, 0.8],
        ])
        self.dpp_ranked = np.array([
            [1, 2, 3],
            [3, 2, 1],
            [3, 1, 2],
            [1, 3, 2],
        ])
        # Set-up
        self.dpp_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0],
            [0, 4, 0, 6, 0, 4, 0, 2, 1, 0, 0, 2, 3, 1, 0, 0],
            [1, 0, 1, 2, 0, 0, 6, 1, 0, 5, 1, 0, 0, 0, 0, 1],
            [0, 2, 3, 0, 0, 0, 0, 0, 4, 1, 3, 2, 0, 0, 1, 4]
        ])

    @patch('btb.recommendation.matrix_factorization.np.random.randint')
    @patch('btb.recommendation.matrix_factorization.NMF')
    def test___init__(self, nmf_mock, randint_mock):
        # set-up
        index = 1
        nmf_mock().fit_transform.return_value = self.dpp_ranked
        randint_mock.return_value = index

        # run
        recommender = MFRecommender(
            self.dpp_matrix,
            self.n_components,
            self.r_min,
        )

        # asserts
        randint_mock.assert_called_once_with(4)  # 4 is self.dpp_matrix length
        np.testing.assert_array_equal(
            recommender.matching_dataset,
            self.dpp_matrix[index, :],
        )
        np.testing.assert_array_equal(recommender.dpp_matrix, self.dpp_matrix)
        np.testing.assert_array_equal(recommender.dpp_ranked, self.dpp_ranked)
        assert recommender.r_minimum == self.r_min

    @patch('btb.recommendation.matrix_factorization.NMF')
    def test_fit(self, nmf_mock):
        nmf_mock().fit_transform.return_value = self.dpp_ranked
        n_components = 3
        dpp_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Run
        recommender = MFRecommender(self.dpp_matrix, n_components, self.r_min)
        i = 0
        nmf_mock().transform.return_value = self.dpp_ranked[i, :]
        recommender.fit(dpp_vector)
        np.testing.assert_array_equal(
            recommender.matching_dataset,
            self.dpp_matrix[i],
        )

    def test_predict_one(self):
        # Run
        recommender = MFRecommender(
            self.dpp_matrix,
            self.n_components,
            self.r_min,
        )
        # matching row is [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0]
        recommender.matching_dataset = self.dpp_matrix[0]

        indicies = [0]
        predictions = recommender.predict(indicies)
        expected = [1]
        np.testing.assert_array_equal(predictions, expected)

    def test_predict_all_matching(self):
        # Run
        recommender = MFRecommender(
            self.dpp_matrix,
            self.n_components,
            self.r_min,
        )
        # matching row is [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0]
        recommender.matching_dataset = self.dpp_matrix[0]
        indicies = [1, 2, 3, 4, 5]
        predictions = recommender.predict(indicies)
        expected = [1, 1, 1, 1, 1]
        np.testing.assert_array_equal(predictions, expected)

    def test_predict_multiple_rankings(self):
        # Run
        recommender = MFRecommender(
            self.dpp_matrix,
            self.n_components,
            self.r_min,
        )
        # matching row is [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0]
        recommender.matching_dataset = self.dpp_matrix[0]
        indicies = [0, 1, 14, 13]
        predictions = recommender.predict(indicies)
        expected = [2, 1, 3, 1]
        np.testing.assert_array_equal(predictions, expected)

    @patch('btb.recommendation.matrix_factorization.UniformRecommender')
    def test_predict_non_zero_rmin(self, uniform_mock):
        # Run
        indicies = [0, 1, 14, 13]
        expected = [3, 1, 2, 1]
        uniform_mock().predict.return_value = expected
        recommender = MFRecommender(
            self.dpp_matrix,
            self.n_components,
            1,
        )
        predictions = recommender.predict(indicies)
        np.testing.assert_array_equal(predictions, expected)

    @patch('btb.recommendation.matrix_factorization.np.random.randint')
    def test_propose_without_add(self, randint_mock):
        index = 0
        randint_mock.return_value = index
        proposed = np.argmax(self.dpp_matrix[0])  # 14
        recommender = MFRecommender(
            self.dpp_matrix,
            self.n_components,
            self.r_min,
        )
        np.testing.assert_array_equal(
            recommender.matching_dataset,
            self.dpp_matrix[index, :],
        )
        pipeline_index = recommender.propose()
        randint_mock.assert_called_once_with(4)  # 4 is self.dpp_matrix length
        assert pipeline_index == proposed

    @patch('btb.recommendation.matrix_factorization.np.random.randint')
    def test_propose_empty_add(self, randint_mock):
        index = 0
        randint_mock.return_value = index
        proposed = np.argmax(self.dpp_matrix[0])  # 14
        recommender = MFRecommender(
            self.dpp_matrix,
            self.n_components,
            self.r_min,
        )
        recommender.add({})
        np.testing.assert_array_equal(
            recommender.matching_dataset,
            self.dpp_matrix[index, :],
        )
        pipeline_index = recommender.propose()
        assert randint_mock.call_count == 2
        assert pipeline_index == proposed
