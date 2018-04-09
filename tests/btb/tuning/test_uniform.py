from unittest import TestCase

import numpy as np
from mock import patch

from btb.hyper_parameter import HyperParameter, ParamTypes
from btb.tuning.uniform import Uniform


class TestUniform(TestCase):

    # METHOD: predict(self):
    # VALIDATE:
    #     * np.random.rand is called with the right values

    @patch('btb.tuning.uniform.np.random')
    def test_predict(self, np_random_mock):

        # Set-up
        np_random_mock.rand.return_value = np.array([.4, .6, .8])

        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = Uniform(tunables)

        # Run
        x = np.array([
            [1., 1],
            [1.2, 2],
            [1.4, 3],
        ])
        predicted = tuner.predict(x)

        # Assert
        expected = np.array([.4, .6, .8])

        np.testing.assert_array_equal(predicted, expected)

        np_random_mock.rand.assert_called_once_with(3, 1)
