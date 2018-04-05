from itertools import product
from unittest import TestCase

import numpy as np
import pytest
from mock import patch

from btb.hyper_parameter import HyperParameter, ParamTypes
from btb.tuning.tuner import BaseTuner


class TestBaseTuner(TestCase):

    # METHOD: __init__(self, tunables, gridding=0, **kwargs)
    # VALIDATE:
    #     * attribute values after creation
    # TODO:
    #     * Remove unnecessary **kwargs

    def test___init__(self):
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )

        # Run
        base_tuner = BaseTuner(tunables, gridding=5)

        # assert
        assert base_tuner.tunables == tunables
        assert base_tuner.grid is True
        assert base_tuner._best_score == -np.inf
        assert base_tuner._best_hyperparams is None
        assert base_tuner.grid_size == 5
        assert base_tuner.X_raw is None
        assert base_tuner.y_raw == []
        assert base_tuner.X.tolist() == []
        assert base_tuner.y.tolist() == []

        expected_grid_axes = [
            np.array([1. , 1.25, 1.5 , 1.75, 2.]),
            np.array([1., 2., 3., 4., 5.])
        ]
        np.testing.assert_array_equal(base_tuner._grid_axes, expected_grid_axes)

    # METHOD: _define_grid(self)
    # VALIDATE:
    #     * grid_axes values
    # TODO:
    #     * return the axes instead of setting an attribute.
    # NOTES:
    #     * Implicitely covered in __init__ method

    # METHOD: _params_to_grid(self, params)
    # VALIDATE:
    #     * Returned grid

    def test__params_to_grid(self):

        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        base_tuner = BaseTuner(tunables, gridding=5)

        # Run
        grid = base_tuner._params_to_grid([1.25, 3])

        # Assert
        np.testing.assert_array_equal(grid, [1, 2])


    # METHOD: _grid_to_params(self, grid_points)
    # VALIDATE:
    #     * Returned params

    def test__grid_to_params(self):

        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        base_tuner = BaseTuner(tunables, gridding=5)

        # Run
        grid = base_tuner._grid_to_params([1, 2])

        # Assert
        np.testing.assert_array_equal(grid, [1.25, 3])


    # METHOD: fit(self, X, y)
    # VALIDATE:
    #     * Set Attributes

    def test_fit(self):

        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        base_tuner = BaseTuner(tunables)

        # Run
        X = [
            [1., 1],
            [1.25, 2],
            [1.5, 3],
            [1.75, 4],
            [2., 5],
        ]
        y = [0.5, 0.6, 0.7, 0.8, 0.9]
        base_tuner.fit(X, y)

        # Assert
        assert base_tuner.X == X
        assert base_tuner.y == y


    # METHOD: _create_candidates(self, n=1000)
    # VALIDATE:
    #     * returned value if self.grid is False: int and float params
    #     * returned value if every point has been used
    #     * returned value if less than n points remain
    #     * returned value if more than n points remain
    #     * BUG: no exception is raised if n > grid_size
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
        )
        base_tuner = BaseTuner(tunables)

        # Run
        candidates = base_tuner._create_candidates(5)

        # Assert
        expected_candidates = np.array([
            [1.0, 1],
            [1.2, 2],
            [1.4, 3],
            [1.6, 4],
            [1.8, 5]
        ])

        np.testing.assert_array_equal(candidates, expected_candidates)

        np_random_mock.rand.assert_called_once_with(5)
        np_random_mock.randint.assert_called_once_with(1, 6, size=5)

    def test__create_candidates_every_point_used(self):
        """every point has been used"""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        base_tuner = BaseTuner(tunables, gridding=5)

        # Insert 25 part_vecs into X
        base_tuner.X = np.array(list(product(*base_tuner._grid_axes)))

        # Run
        candidates = base_tuner._create_candidates(5)

        # Assert
        assert candidates is None

    def test__create_candidates_lt_n_remaining(self):
        """less than n points remaining"""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        base_tuner = BaseTuner(tunables, gridding=5)

        # Insert 20 part_vecs into X (5 remaining)
        all_vecs = np.array(list(product(*base_tuner._grid_axes)))
        base_tuner.X = all_vecs[:20]

        # Run
        # n = 1000
        candidates = base_tuner._create_candidates()

        # Assert
        expected_candidates = all_vecs[20:]
        np.testing.assert_array_equal(np.sort(candidates, axis=0), expected_candidates)

    @patch('btb.tuning.tuner.np.random')
    def test__create_candidates_gt_n_remaining(self, np_random_mock):
        """more than n points remaining"""
        # Set-up
        np_random_mock.randint.side_effect = [
            np.array([0, 0]),   # [1.0, 1] => Used
            np.array([1, 1]),
            np.array([2, 2]),   # [1.4, 3] => Used
            np.array([3, 3]),
            np.array([4, 4]),
        ]

        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        base_tuner = BaseTuner(tunables, gridding=5)

        # Insert 2 vectors that we assume as used
        base_tuner.X = np.array([
            [1.0, 1],
            [1.5, 3]
        ])

        # Run
        candidates = base_tuner._create_candidates(3)

        # Assert
        expected_candidates = np.array([
            [1.25, 2],
            [1.75, 4],
            [2., 5]
        ])
        np.testing.assert_array_equal(candidates, expected_candidates)

    # METHOD: predict(self, X)
    # VALIDATE:
    #     * Exception is raised
    def test__predict(self):
        """Exception is raised"""
        base_tuner = BaseTuner(tuple())

        with pytest.raises(NotImplementedError):
            base_tuner.predict([])

    # METHOD: _acquire(self, predictions)
    # VALIDATE:
    #     * returned value
    # NOTES:
    #     * Implictely covered in propose method
    #     * Fails with multi-dimensional ndarrays
    def test___acquire(self):
        """np.argmax is properly called"""
        # Set-up
        base_tuner = BaseTuner(tuple())

        # Run
        predictions = np.array([0.9, 0.95, 0.8])
        idx = base_tuner._acquire(predictions)

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
        base_tuner = BaseTuner(tunables)

        create_candidates_mock.return_value = None

        # Run
        params = base_tuner.propose(1)

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
        base_tuner = BaseTuner(tunables)

        create_candidates_mock.return_value = np.array([
            [1.0, 1],
            [1.2, 2],
            [1.4, 3],
            [1.6, 4],
            [1.8, 5]
        ])

        predict_mock.return_value = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        # Run
        params = base_tuner.propose(1)

        # Assert
        expected_params = {'a_float_param': 1.8, 'an_int_param': 5}
        assert params == expected_params

    @patch('btb.tuning.tuner.BaseTuner._create_candidates')
    @patch('btb.tuning.tuner.BaseTuner.predict')
    def test_propse_n_gt_1(self, predict_mock, create_candidates_mock):
        """n == 1"""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        base_tuner = BaseTuner(tunables)

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
        params = base_tuner.propose(3)

        # Assert
        expected_params = [
            {'a_float_param': 1.4, 'an_int_param': 3},
            {'a_float_param': 1.6, 'an_int_param': 4},
            {'a_float_param': 1.8, 'an_int_param': 5},
        ]
        assert params == expected_params

    # METHOD: add(self, X, y)
    # VALIDATE:
    #     * Test attribute values after
    # TODO:
    #     * Split this method in smaller ones
