from unittest import TestCase

import numpy as np
from mock import patch

from btb.recommendation.recommender import BaseRecommender


class TestBaseRecommender(TestCase):
    def setUp(self):
        self.n_components = 3
        # Set-up
        self.dpp_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 3, 0, 4, 6, 2, 5, 0, 8, 0],
            [0, 4, 0, 6, 0, 4, 0, 2, 1, 0, 0, 2, 3, 1, 0, 0],
            [1, 0, 1, 2, 0, 0, 6, 1, 0, 5, 1, 0, 0, 0, 0, 1],
            [0, 2, 3, 0, 0, 0, 0, 0, 4, 1, 3, 2, 0, 0, 1, 4]
        ])

    def test___init__(self):
        # Run
        recommender = BaseRecommender(self.dpp_matrix)
        np.testing.assert_array_equal(recommender.dpp_matrix, self.dpp_matrix)
        # assert dpp_vector has same number of entries as pipelines
        assert recommender.dpp_vector.shape[0] == self.dpp_matrix.shape[1]

    def test_fit(self):
        X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Run
        recommender = BaseRecommender(self.dpp_matrix)
        recommender.fit(X)
        np.testing.assert_array_equal(
            X,
            recommender.dpp_vector,
        )

    def test__get_candidates_all(self):
        recommender = BaseRecommender(self.dpp_matrix)
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
        recommender = BaseRecommender(self.dpp_matrix)
        X = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0])
        recommender.dpp_vector = X
        expected = np.array([0, 1, 2, 4, 6, 8, 10, 11, 12, 14, 15])
        candidates = recommender._get_candidates()
        np.testing.assert_array_equal(
            candidates,
            expected,
        )

    def test__get_candidates_none(self):
        recommender = BaseRecommender(self.dpp_matrix)
        X = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        recommender.dpp_vector = X
        candidates = recommender._get_candidates()
        expected = None
        self.assertEqual(expected, candidates)

    def test__acquire_last(self):
        recommender = BaseRecommender(self.dpp_matrix)
        predictions = np.array([1, 3, 5, 7])
        acquired = recommender._acquire(predictions)
        expected = 3
        self.assertEqual(acquired, expected)

    def test__acquire_middle(self):
        recommender = BaseRecommender(self.dpp_matrix)
        predictions = np.array([9, 10, 5, 7])
        acquired = recommender._acquire(predictions)
        expected = 1
        self.assertEqual(acquired, expected)

    def test__acquire_multiple(self):
        recommender = BaseRecommender(self.dpp_matrix)
        predictions = np.array([1, 9, 9, 7])
        acquired = recommender._acquire(predictions)
        expected_1 = 2
        expected_2 = 1
        self.assertTrue(acquired == expected_1 or acquired == expected_2)

    def test_predict(self):
        # Run
        recommender = BaseRecommender(self.dpp_matrix)
        indicies = [0]
        self.assertRaises(NotImplementedError, recommender.predict, indicies)

    @patch('btb.recommendation.recommender.BaseRecommender.fit')
    def test_add_none(self, fit_mock):
        recommender = BaseRecommender(self.dpp_matrix)
        recommender.add({})
        expected_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(recommender.dpp_vector, expected_x)
        np.testing.assert_array_equal(
            fit_mock.call_args[0][0],
            expected_x.reshape(1, -1)
        )
        fit_mock.assert_called_once()

    @patch('btb.recommendation.recommender.BaseRecommender.fit')
    def test_add_once(self, fit_mock):
        recommender = BaseRecommender(self.dpp_matrix)
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

    @patch('btb.recommendation.recommender.BaseRecommender.fit')
    def test_add_twice(self, fit_mock):
        recommender = BaseRecommender(self.dpp_matrix)
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

    @patch('btb.recommendation.recommender.BaseRecommender._get_candidates')
    def test_propose_done(self, get_candidates_mock):
        # Set-up
        recommender = BaseRecommender(self.dpp_matrix)

        get_candidates_mock.return_value = None

        # Run
        params = recommender.propose()

        # Assert
        expected_params = None
        assert params == expected_params

    @patch('btb.recommendation.recommender.BaseRecommender._get_candidates')
    @patch('btb.recommendation.recommender.BaseRecommender.predict')
    def test_propose_once(self, predict_mock, get_candidates_mock):
        """n == 1"""
        # Set-up
        recommender = BaseRecommender(self.dpp_matrix)

        get_candidates_mock.return_value = [1, 3, 7, 8]

        predict_mock.return_value = np.array([1, 4, 2, 3])

        # Run
        pipeline_index = recommender.propose()

        # Assert
        expected_pipeline_index = 3
        assert pipeline_index == expected_pipeline_index
