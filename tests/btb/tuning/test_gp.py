from unittest import TestCase

import numpy as np
from mock import Mock, patch

from btb.hyper_parameter import HyperParameter, ParamTypes
from btb.tuning.gp import GP, GPEi, GPEiVelocity


class TestGP(TestCase):

    # METHOD __init__(self, tunables, gridding=0, **kwargs)
    # VALIDATE:
    #     * attribute values

    def test___init__(self):
        # Run
        tuner = GP(tuple(), r_minimum=5)

        # assert
        assert tuner.r_minimum == 5

    # METHOD: fit(self, X, y)
    # VALIDATE:
    #     * if X shorter than r_minimum, nothing is done
    #     * GaussianProcessRegressor is called with the right values.
    # NOTES:
    #     * GPR will need to be mocked.

    @patch('btb.tuning.gp.GaussianProcessRegressor')
    def test_fit_lt_r_min(self, gpr_mock):
        """If the length of X is smaller than r_minimum, nothing is done."""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = GP(tunables, r_minimum=5)

        # Run
        X = np.array([
            [1., 1],
            [1.2, 2],
            [1.4, 4]
        ])
        y = np.array([0.5, 0.6, 0.7])
        tuner.fit(X, y)

        # assert
        np.testing.assert_array_equal(tuner.X, X)
        np.testing.assert_array_equal(tuner.y, y)
        gpr_mock.assert_not_called()

    @patch('btb.tuning.gp.GaussianProcessRegressor')
    def test_fit_gt_r_min(self, gpr_mock):
        """If the length of X is greater than r_minimum, gpr is fit."""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = GP(tunables, r_minimum=2)

        gp_mock = Mock()
        gpr_mock.return_value = gp_mock

        # Run
        X = np.array([
            [1., 1],
            [1.2, 2],
            [1.4, 4]
        ])
        y = np.array([0.5, 0.6, 0.7])
        tuner.fit(X, y)

        # assert
        np.testing.assert_array_equal(tuner.X, X)
        np.testing.assert_array_equal(tuner.y, y)
        gpr_mock.assert_called_once_with(normalize_y=True)
        gp_mock.fit.assert_called_once_with(X, y)

    # METHOD: predict(self, X)
    # VALIDATE:
    #     * if X shorter than r_minimum, Uniform is used.
    #     * GPR is colled with the right values
    # NOTES:
    #     * GPR will need to be mocked

    @patch('btb.tuning.gp.Uniform')
    def test_predict_x_lt_r_min(self, uniform_mock):
        """If the length of self.X is smaller than r_minimum, Uniform is used."""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = GP(tunables, r_minimum=5)

        tuner.X = np.array([
            [1., 1],
            [1.2, 2],
            [1.4, 4]
        ])

        tuner.gp = Mock()

        predict_mock = Mock()
        predict_mock.return_value = np.array([0.8, 0.9])
        uniform_instance_mock = Mock()
        uniform_instance_mock.predict = predict_mock
        uniform_mock.return_value = uniform_instance_mock

        # Run
        X = np.array([
            [1.6, 4],
            [1.8, 5],
        ])
        predicted = tuner.predict(X)

        # assert
        expected = np.array([0.8, 0.9])
        np.testing.assert_array_equal(predicted, expected)
        uniform_mock.assert_called_once_with(tunables)
        predict_mock.assert_called_once_with(X)
        tuner.gp.predict.assert_not_called()

    @patch('btb.tuning.gp.Uniform')
    def test_predict_x_gt_r_min(self, uniform_mock):
        """If the length of self.X is greater than r_minimum, self.gp is used."""
        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = GP(tunables, r_minimum=2)

        tuner.X = np.array([
            [1., 1],
            [1.2, 2],
            [1.4, 4]
        ])

        tuner.gp = Mock()
        tuner.gp.predict.return_value = (
            np.array([0.8, 0.9]),
            np.array([0.1, 0.3])
        )

        # Run
        X = np.array([
            [1.6, 4],
            [1.8, 5],
        ])
        predicted = tuner.predict(X)

        # assert
        expected = np.array([
            [0.8, 0.1],
            [0.9, 0.3]
        ])
        np.testing.assert_array_equal(predicted, expected)
        tuner.gp.predict.assert_called_once_with(X, return_std=True)

    # METHOD: _acquire(self, predictions)
    # VALIDATE:
    #     * returned values

    def test__acquire(self):

        # Set-up
        tuner = GP(tuple(), r_minimum=2)

        # Run
        predictions = np.array([
            [0.8, 0.1],
            [0.9, 0.3]
        ])
        best = tuner._acquire(predictions)

        # assert
        assert best == 1


class TestGPEi(TestCase):

    # METHOD: _acquire(self, predictions)
    # VALIDATE:
    #     * return values according to the formula

    def test__acquire_best_ei_eq_best_y(self):
        """Best Expected Improvement corresponds to the best prediction."""

        # Set-up
        tuner = GPEi(tuple(), r_minimum=2)
        tuner.y = np.array([0.5, 0.6, 0.7])

        # Run
        predictions = np.array([
            [0.8, 1],
            [0.9, 2]
        ])
        best = tuner._acquire(predictions)

        # assert
        assert best == 1

    def test__acquire_best_ei_neq_best_y(self):
        """Best Expected Improvement does NOT correspond to the best prediction."""

        # Set-up
        tuner = GPEi(tuple(), r_minimum=2)
        tuner.y = np.array([0.5, 0.6, 0.7])

        # Run
        predictions = np.array([
            [0.8, 2],
            [0.9, 1]
        ])
        best = tuner._acquire(predictions)

        # assert
        assert best == 0

    def test__acquire_possible_error(self):
        """Manually crafted case that seems to be an error in the formula:

        The second prediction has a higher score and both have the same stdev.
        However, the formula indicates that the first prediction is the best one.
        """

        # Set-up
        tuner = GPEi(tuple(), r_minimum=2)
        tuner.y = np.array([0.5, 0.6, 0.7])

        # Run
        predictions = np.array([
            [0.8, 1],
            [0.9, 1]
        ])
        best = tuner._acquire(predictions)

        # assert
        assert best == 0


class TestGPEiVelocity(TestCase):

    # METHOD: fit(self, X, y)
    # VALIDATE:
    #     * if y shorter than r_minimum, nothing is done
    #     * POU attribute values according to the formula

    @patch('btb.tuning.gp.GPEi.fit')
    def test_fit_lt_r_min(self, fit_mock):
        """If the length of X is smaller than r_minimum, nothing is done."""
        # Set-up
        tuner = GPEiVelocity(tuple(), r_minimum=5)

        # Run
        X = np.array([
            [1., 1],
            [1.2, 2],
            [1.4, 4]
        ])
        y = np.array([0.5, 0.6, 0.7])
        tuner.fit(X, y)

        # assert
        assert tuner.POU == 0

    @patch('btb.tuning.gp.GPEi.fit')
    def test_fit_gt_r_min(self, fit_mock):
        """If the length of X is greater than r_minimum, calculate POU."""
        # Set-up
        tuner = GPEiVelocity(tuple(), r_minimum=2)

        # Run
        X = np.array([
            [1., 1],
            [1.2, 2],
            [1.4, 3],
            [1.6, 4],
            [1.8, 5]
        ])
        y = np.array([0.8, 0.81, 0.9, 0.84, 0.87])
        tuner.fit(X, y)

        # assert
        assert tuner.POU == 0.08208499862389883

    # METHOD: predict(self, X, y)
    # VALIDATE:
    #     * cases where Uniform is returned
    # NOTES:
    #     * random will need to be mocked

    @patch('btb.tuning.gp.Uniform')
    @patch('btb.tuning.gp.np.random')
    @patch('btb.tuning.gp.GPEi.predict')
    def test_fit_predict_uniform(self, predict_mock, random_mock, uniform_mock):
        """If random is lower than POU, Uniform is used."""

        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = GPEiVelocity(tunables, r_minimum=2)

        tuner.POU = 0.05

        random_mock.random.return_value = 0.01

        uniform_instance_mock = Mock()
        uniform_mock.return_value = uniform_instance_mock

        # Run
        X = np.array([
            [1., 1],
            [1.2, 2],
            [1.4, 3],
            [1.6, 4],
            [1.8, 5]
        ])
        tuner.predict(X)

        # assert
        uniform_mock.assert_called_once_with(tunables)
        uniform_instance_mock.predict.assert_called_once_with(X)

        predict_mock.assert_not_called()

    @patch('btb.tuning.gp.Uniform')
    @patch('btb.tuning.gp.np.random')
    @patch('btb.tuning.gp.GPEi.predict')
    def test_fit_predict_super(self, predict_mock, random_mock, uniform_mock):
        """If random is higher than POU, super.predict is used."""

        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = GPEiVelocity(tunables, r_minimum=2)

        tuner.POU = 0.05

        random_mock.random.return_value = 0.1

        # Run
        X = np.array([
            [1., 1],
            [1.2, 2],
            [1.4, 3],
            [1.6, 4],
            [1.8, 5]
        ])
        tuner.predict(X)

        # assert
        predict_mock.assert_called_once_with(X)
        uniform_mock.assert_not_called()
