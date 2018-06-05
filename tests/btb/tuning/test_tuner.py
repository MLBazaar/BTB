from itertools import product
from unittest import TestCase

import numpy as np
import pytest
from mock import call, patch
from numpy.random import shuffle as np_shuffle

from btb.hyper_parameter import HyperParameter, ParamTypes
from btb.tuning.tuner import BaseTuner


class TestBaseTuner(TestCase):

    # METHOD: __init__(self, tunables, gridding=0, **kwargs)
    # VALIDATE:
    #     * attribute values after creation

    def test___init__(self):
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )

        # Run
        tuner = BaseTuner(tunables, gridding=5)

        # assert
        assert tuner.tunables == tunables
        assert tuner.grid is True
        assert tuner._best_score == -np.inf
        assert tuner._best_hyperparams is None
        assert tuner.grid_width == 5
        assert tuner.X_raw is None
        assert tuner.y_raw == []
        assert tuner.X.tolist() == []
        assert tuner.y.tolist() == []

        expected_grid_axes = [
            np.array([1., 1.25, 1.5, 1.75, 2.]),
            np.array([1., 2., 3., 4., 5.])
        ]
        np.testing.assert_array_equal(tuner._grid_axes, expected_grid_axes)

    # METHOD: _generate_grid(self)
    # VALIDATE:
    #     * grid_axes values
    # NOTES:
    #     * Implicitely covered in __init__ method

    # METHOD: fit(self, X, y)
    # VALIDATE:
    #     * Set Attributes

    def test_fit(self):

        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = BaseTuner(tunables)

        # Run
        X = [
            [1., 1],
            [1.25, 2],
            [1.5, 3],
            [1.75, 4],
            [2., 5],
        ]
        y = [0.5, 0.6, 0.7, 0.8, 0.9]
        tuner.fit(X, y)

        # Assert
        assert tuner.X == X
        assert tuner.y == y

    # METHOD: _create_candidates(self, n=1000)
    # VALIDATE:
    #     * returned value if self.grid is False: int and float params
    #     * returned value if every point has been used
    #     * returned value if less than n points remain
    #     * returned value if more than n points remain
    #     * BUG: no exception is raised if n > grid_size
    #     * BUG: GH74: STRING ParamTypes do not work
    # TODO:
    #     * Split this method in 4 smaller methods

    @patch('btb.tuning.tuner.np.random')
    def test__create_candidates_no_grid(self, np_random_mock):
        """self.grid is False"""
        # Set-up
        np_random_mock.rand.return_value = np.array([.0, .2, .4, .6, .8])
        np_random_mock.randint.return_value = np.array([1, 2, 3, 4, 5])

        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
            ('a_string_param', HyperParameter(ParamTypes.STRING, ['a', 'b', 'c'])),
        )
        tuner = BaseTuner(tunables)

        # Run
        candidates = tuner._create_candidates(5)

        # Assert
        expected_candidates = np.array([
            [1.0, 1, 0.],
            [1.2, 2, 0.2],
            [1.4, 3, 0.4],
            [1.6, 4, 0.6],
            [1.8, 5, 0.8]
        ])

        np.testing.assert_array_equal(candidates, expected_candidates)

        expected_calls = [
            call(5),
            call(5),
        ]
        np_random_mock.rand.assert_has_calls(expected_calls)
        np_random_mock.randint.assert_called_once_with(1, 6, size=5)

    def test__create_candidates_every_point_used(self):
        """every point has been used"""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = BaseTuner(tunables, gridding=5)

        # Insert 25 part_vecs into X
        tuner.X = np.array(list(product(*tuner._grid_axes)))

        # Run
        candidates = tuner._create_candidates(5)

        # Assert
        assert candidates is None

    @patch('btb.tuning.tuner.np.random')
    def test__create_candidates_lt_n_remaining(self, np_random_mock):
        """less than n points remaining"""
        # Set-up
        mock_context = dict()

        def shuffle(array):
            np_shuffle(array)

            # Store a copy of the array for the assert
            mock_context['shuffled_array'] = array.copy()

        np_random_mock.shuffle.side_effect = shuffle

        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = BaseTuner(tunables, gridding=3)

        # Insert 5 used vectors into X (4 remaining)
        tuner.X = np.array([
            [1., 1.],
            [1., 3.],
            [1., 5.],
            [1.5, 1.],
            [1.5, 5.],
        ])

        # Run
        # n = 1000
        candidates = tuner._create_candidates()

        # Assert
        expected_candidates = mock_context['shuffled_array']
        np.testing.assert_array_equal(candidates, expected_candidates)

    @patch('btb.tuning.tuner.np.random')
    def test__create_candidates_gt_n_remaining(self, np_random_mock):
        """more than n points remaining"""
        # Set-up
        mock_context = dict()

        def shuffle(array):
            np_shuffle(array)

            # Store a copy of the array for the assert
            mock_context['shuffled_array'] = array.copy()

        np_random_mock.shuffle.side_effect = shuffle

        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = BaseTuner(tunables, gridding=3)

        # Insert 5 used vectors into X (4 remaining)
        tuner.X = np.array([
            [1., 1.],
            [1., 3.],
            [1., 5.],
            [1.5, 1.],
            [1.5, 5.],
        ])

        # Run
        candidates = tuner._create_candidates(2)

        # Assert
        expected_candidates = mock_context['shuffled_array'][0:2]
        np.testing.assert_array_equal(candidates, expected_candidates)

    # METHOD: predict(self, X)
    # VALIDATE:
    #     * Exception is raised
    def test__predict(self):
        """Exception is raised"""
        tuner = BaseTuner(tuple())

        with pytest.raises(NotImplementedError):
            tuner.predict([])

    # METHOD: _acquire(self, predictions)
    # VALIDATE:
    #     * returned value
    # NOTES:
    #     * Implictely covered in propose method
    #     * Fails with multi-dimensional ndarrays
    def test___acquire(self):
        """np.argmax is properly called"""
        # Set-up
        tuner = BaseTuner(tuple())

        # Run
        predictions = np.array([0.9, 0.95, 0.8])
        idx = tuner._acquire(predictions)

        # Assert
        assert idx == 1

    # METHOD: propose(self, n=1)
    # VALIDATE:
    #     * Returned values if gridding is done
    #     * Returned values if n == 1
    #     * Returned values if n != 1
    #     * self.predct is called
    # NOTES:
    #     * self.predict will need to be mocked to prevent the NotImplemented Exception
    # TODO:
    #     * Instead calling create_candidates N times and get the best one each time
    #       Why not calling it 1 time and getting the N best predictions?

    @patch('btb.tuning.tuner.BaseTuner._create_candidates')
    @patch('btb.tuning.tuner.BaseTuner.predict')
    def test_propse_done(self, predict_mock, create_candidates_mock):
        """gridding is done"""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = BaseTuner(tunables)

        create_candidates_mock.return_value = None

        # Run
        params = tuner.propose(1)

        # Assert
        expected_params = None
        assert params == expected_params

    @patch('btb.tuning.tuner.BaseTuner._create_candidates')
    @patch('btb.tuning.tuner.BaseTuner.predict')
    def test_propse_n_eq_1(self, predict_mock, create_candidates_mock):
        """n == 1"""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = BaseTuner(tunables)

        create_candidates_mock.return_value = np.array([
            [1.0, 1],
            [1.2, 2],
            [1.4, 3],
            [1.6, 4],
            [1.8, 5]
        ])

        predict_mock.return_value = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        # Run
        params = tuner.propose(1)

        # Assert
        expected_params = {'a_float_param': 1.8, 'an_int_param': 5}
        assert params == expected_params

    @patch('btb.tuning.tuner.BaseTuner._create_candidates')
    @patch('btb.tuning.tuner.BaseTuner.predict')
    def test_propse_n_gt_1(self, predict_mock, create_candidates_mock):
        """n > 1"""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = BaseTuner(tunables)

        create_candidates_mock.side_effect = [
            np.array([[1.0, 1], [1.4, 3]]),
            np.array([[1.2, 2], [1.6, 4]]),
            np.array([[1.6, 4], [1.8, 5]]),
        ]

        predict_mock.side_effect = [
            np.array([0.5, 0.7]),
            np.array([0.6, 0.8]),
            np.array([0.7, 0.9]),
        ]

        # Run
        params = tuner.propose(3)

        # Assert
        expected_params = [
            {'a_float_param': 1.4, 'an_int_param': 3},
            {'a_float_param': 1.6, 'an_int_param': 4},
            {'a_float_param': 1.8, 'an_int_param': 5},
        ]
        assert params == expected_params

    # METHOD: add(self, X, y)
    # VALIDATE:
    #     * Test attribute values after the call
    # TODO:
    #     * Split this method in smaller ones
    #     * Why is there an if in the x_transformed part?
    #     * Use "for Xi, yi in zip(X, y)" instead of "for i in range(len(X))"

    def test_add(self):

        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT_EXP, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = BaseTuner(tunables)

        # Run
        # first run
        X = {'a_float_param': 1., 'an_int_param': 1}
        y = 0.5
        tuner.add(X, y)

        # second run (test append)
        X = [
            {'a_float_param': 1.2, 'an_int_param': 2},
            {'a_float_param': 1.4, 'an_int_param': 3}
        ]
        y = [0.6, 0.7]
        tuner.add(X, y)

        # Assert
        expected_X = np.array([
            [0., 1],
            [0.07918124604762482, 2],
            [0.146128035678238, 3],
        ])
        expected_X_raw = np.array([
            [1.0, 1],
            [1.2, 2],
            [1.4, 3],
        ])
        expected_y = np.array([0.5, 0.6, 0.7])

        np.testing.assert_array_equal(tuner.X, expected_X)
        np.testing.assert_array_equal(tuner.X_raw, expected_X_raw)
        np.testing.assert_array_equal(tuner.y, expected_y)
        np.testing.assert_array_equal(tuner.y_raw, expected_y)
