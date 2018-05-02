from unittest import TestCase

import numpy as np
from mock import patch

from btb.recommendation.recommender import Recommender


class TestBaseRecommender(TestCase):
    def setUp(self):
        self.n_components = 3
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

    @patch('btb.recommendation.recommender.np.random.randint')
    @patch('btb.recommendation.recommender.NMF')
    def test___init__(self, nmf_mock, randint_mock):
        # set-up
        index = 1
        nmf_mock().fit_transform.return_value = self.dpp_ranked
        randint_mock.return_value = index

        # run
        recommender = Recommender(self.dpp_matrix, self.n_components)

        # asserts
        randint_mock.assert_called_once_with(4)  # 4 is self.dpp_matrix length
        np.testing.assert_array_equal(
            recommender.matching_dataset,
            self.dpp_matrix[index, :],
        )
        np.testing.assert_array_equal(recommender.dpp_matrix, self.dpp_matrix)
        np.testing.assert_array_equal(recommender.dpp_ranked, self.dpp_ranked)

    @patch('btb.recommendation.recommender.NMF')
    def test_fit(self, nmf_mock):
        nmf_mock().fit_transform.return_value = self.dpp_ranked
        n_components = 3
        X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Run
        recommender = Recommender(self.dpp_matrix, n_components)
        for i in range(self.dpp_ranked.shape[0]):
            nmf_mock().transform.return_value = self.dpp_ranked[i, :]
            recommender.fit(X)
            np.testing.assert_array_equal(
                recommender.matching_dataset,
                self.dpp_matrix[i],
            )

    def test__get_candidates_all(self):
        recommender = Recommender(self.dpp_matrix, self.n_components)
        X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        recommender.dpp_vector = X
        candidates = recommender._get_candidates()
        expected = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        )
        np.testing.assert_array_equal(
            candidates,
            expected,
        )

    def test__get_candidates_some(self):
        recommender = Recommender(self.dpp_matrix, self.n_components)
        X = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0])
        recommender.dpp_vector = X
        expected = np.array([0, 1, 2, 4, 6, 8, 10, 11, 12, 14, 15])
        candidates = recommender._get_candidates()
        np.testing.assert_array_equal(
            candidates,
            expected,
        )

    def test__get_candidates_none(self):
        recommender = Recommender(self.dpp_matrix, self.n_components)
        X = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        recommender.dpp_vector = X
        candidates = recommender._get_candidates()
        expected = None
        self.assertEqual(expected, candidates)

    def test__acquire_last(self):
        recommender = Recommender(self.dpp_matrix, self.n_components)
        predictions = np.array([1, 3, 5, 7])
        acquired = recommender._acquire(predictions)
        expected = 3
        self.assertEqual(acquired, expected)

    def test__acquire_middle(self):
        recommender = Recommender(self.dpp_matrix, self.n_components)
        predictions = np.array([9, 10, 5, 7])
        acquired = recommender._acquire(predictions)
        expected = 1
        self.assertEqual(acquired, expected)

    def test__acquire_multiple(self):
        recommender = Recommender(self.dpp_matrix, self.n_components)
        predictions = np.array([1, 9, 9, 7])
        acquired = recommender._acquire(predictions)
        expected_1 = 2
        expected_2 = 1
        self.assertTrue(acquired == expected_1 or acquired == expected_2)

    def test_predict_one(self):
        # Run
        recommender = Recommender(self.dpp_matrix, self.n_components)
        # matching row is [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0]
        recommender.matching_dataset = self.dpp_matrix[0]

        indicies = [0]
        predictions = recommender.predict(indicies)
        expected = [1]
        np.testing.assert_array_equal(predictions, expected)

    def test_predict_all_matching(self):
        # Run
        recommender = Recommender(self.dpp_matrix, self.n_components)
        # matching row is [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0]
        recommender.matching_dataset = self.dpp_matrix[0]
        indicies = [1, 2, 3, 4, 5]
        predictions = recommender.predict(indicies)
        expected = [1, 1, 1, 1, 1]
        np.testing.assert_array_equal(predictions, expected)

    def test_predict_multiple_rankings(self):
        # Run
        recommender = Recommender(self.dpp_matrix, self.n_components)
        # matching row is [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0]
        recommender.matching_dataset = self.dpp_matrix[0]
        indicies = [0, 1, 14, 13]
        predictions = recommender.predict(indicies)
        expected = [2, 1, 3, 1]
        np.testing.assert_array_equal(predictions, expected)

    @patch('btb.recommendation.recommender.Recommender.fit')
    def test_add_none(self, fit_mock):
        recommender = Recommender(self.dpp_matrix, self.n_components)
        recommender.add({})
        expected_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(recommender.dpp_vector, expected_x)
        np.testing.assert_array_equal(
            fit_mock.call_args[0][0],
            expected_x.reshape(1, -1)
        )
        fit_mock.assert_called_once()

    @patch('btb.recommendation.recommender.Recommender.fit')
    def test_add_once(self, fit_mock):
        recommender = Recommender(self.dpp_matrix, self.n_components)
        recommender.add({1: 2, 3: 4, 5: 1})
        expected_x_1 = np.array(
            [0, 2, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        np.testing.assert_array_equal(recommender.dpp_vector, expected_x_1)
        np.testing.assert_array_equal(
            fit_mock.call_args[0][0],
            expected_x_1.reshape(1, -1)
        )
        fit_mock.assert_called_once()

    @patch('btb.recommendation.recommender.Recommender.fit')
    def test_add_twice(self, fit_mock):
        recommender = Recommender(self.dpp_matrix, self.n_components)
        recommender.add({1: 2, 3: 4, 5: 1})
        expected_x_1 = np.array(
            [0, 2, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        np.testing.assert_array_equal(recommender.dpp_vector, expected_x_1)
        np.testing.assert_array_equal(
            fit_mock.call_args[0][0],
            expected_x_1.reshape(1, -1)
        )
        recommender.add({1: 1, 6: 3, 9: 4})
        expected_x_2 = np.array(
            [0, 1, 0, 4, 0, 1, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0]
        )
        np.testing.assert_array_equal(recommender.dpp_vector, expected_x_2)
        np.testing.assert_array_equal(
            fit_mock.call_args[0][0],
            expected_x_2.reshape(1, -1)
        )
        self.assertEqual(fit_mock.call_count, 2)

    @patch('btb.recommendation.recommender.Recommender._get_candidates')
    @patch('btb.recommendation.recommender.Recommender.predict')
    def test_propose_done(self, predict_mock, get_candidates_mock):
        # Set-up
        recommender = Recommender(self.dpp_matrix, self.n_components)

        get_candidates_mock.return_value = None

        # Run
        params = recommender.propose()

        # Assert
        expected_params = None
        assert params == expected_params

    @patch('btb.recommendation.recommender.Recommender._get_candidates')
    @patch('btb.recommendation.recommender.Recommender.predict')
    def test_propose_once(self, predict_mock, get_candidates_mock):
        """n == 1"""
        # Set-up
        recommender = Recommender(self.dpp_matrix, self.n_components)

        get_candidates_mock.return_value = [1, 3, 7, 8]

        predict_mock.return_value = np.array([1, 4, 2, 3])

        # Run
        pipeline_index = recommender.propose()

        # Assert
        expected_pipeline_index = 3
        assert pipeline_index == expected_pipeline_index

    @patch('btb.recommendation.recommender.np.random.randint')
    def test_propose_without_add(self, randint_mock):
        index = 0
        randint_mock.return_value = index
        proposed = np.argmax(self.dpp_matrix[0])  # 14
        recommender = Recommender(self.dpp_matrix, self.n_components)
        np.testing.assert_array_equal(
            recommender.matching_dataset,
            self.dpp_matrix[index, :],
        )
        pipeline_index = recommender.propose()
        randint_mock.assert_called_once_with(4)  # 4 is self.dpp_matrix length
        assert pipeline_index == proposed

    @patch('btb.recommendation.recommender.np.random.randint')
    def test_propose_empty_add(self, randint_mock):
        index = 0
        randint_mock.return_value = index
        proposed = np.argmax(self.dpp_matrix[0])  # 14
        recommender = Recommender(self.dpp_matrix, self.n_components)
        recommender.add({})
        np.testing.assert_array_equal(
            recommender.matching_dataset,
            self.dpp_matrix[index, :],
        )
        pipeline_index = recommender.propose()
        assert randint_mock.call_count == 2
        assert pipeline_index == proposed
