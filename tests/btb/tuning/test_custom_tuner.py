from unittest import TestCase

import numpy as np
from mock import patch

from btb.hyper_parameter import HyperParameter, ParamTypes
from btb.tuning.custom_tuner import CustomTuner


class TestCustomTuner(TestCase):

    # METHOD: propose(self):
    # VALIDATE:
    #     * self._create_candidates is called with the right values
    # NOTES:
    #     * If _create_candidates returns None (gridding finished) it will fail.
    #     * Is the returned value right? Shouldn't it be a dict?

    @patch('btb.tuning.tuner.BaseTuner._create_candidates')
    def test_propse(self, create_candidates_mock):

        # Set-up
        tunables = (
            ('a_float_param', HyperParameter(ParamTypes.FLOAT, [1., 2.])),
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 5])),
        )
        tuner = CustomTuner(tunables)

        create_candidates_mock.return_value = np.array([[1.0, 1]])

        # Run
        params = tuner.propose()

        # Assert
        expected_params = np.array([1.0, 1])
        np.testing.assert_array_equal(params, expected_params)

        create_candidates_mock.assert_called_once_with(1)
