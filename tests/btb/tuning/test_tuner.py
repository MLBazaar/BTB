from unittest import TestCase

import numpy as np
from btb.hyper_parameter import HyperParameter, ParamTypes
from btb.tuning.tuner import BaseTuner


class TestBaseTuner(TestCase):

    def setUp(self):
        self.tunables = (
            ('an_int_param', HyperParameter(ParamTypes.INT, [1, 10])),
        )
        self.base_tuner = BaseTuner(self.tunables, gridding=5)

    # METHOD: __init__(self, tunables, gridding=0, **kwargs)
    # VALIDATE:
    #     * attribute values after creation
    def test___init__(self):

        assert self.base_tuner.tunables == self.tunables
        assert self.base_tuner.grid == True
        assert self.base_tuner._best_score == -np.inf
        assert self.base_tuner._best_hyperparams == None
        assert self.base_tuner.grid_size == 5
        assert len(self.base_tuner._grid_axes) == 1
        assert self.base_tuner._grid_axes[0].tolist() == [1.,  3.,  6.,  8., 10.]
        assert self.base_tuner.X_raw == None
        assert self.base_tuner.y_raw == []
        assert self.base_tuner.X.tolist() == []
        assert self.base_tuner.y.tolist() == []

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


    # METHOD: _grid_to_params(self, grid_points)
    # VALIDATE:
    #     * Returned params

    # METHOD: fit(self, X, y)
    # VALIDATE:
    #     * Set Attributes

    # METHOD: _create_candidates(self, n=1000)
    # VALIDATE:
    #     * returned value if every point has been used
    #     * returned value if less than n points remain
    #     * returned value if more than n points remain
    #     * returned value if self.grid is False
    # TODO:
    #     * Split this method in 4 smaller methods

    # METHOD: predict(self, X)
    # VALIDATE:
    #     * Exception is raised

    # METHOD: _acquire(self, predictions)
    # VALIDATE:
    #     * np.argmax is called
    # NOTES:
    #     * Implictely covered in propose method

    # METHOD: propose(self, n=1)
    # VALIDATE:
    #     * Returned values if n == 1
    #     * Returned values if n != 1
    #     * self.predct is called
    # NOTES:
    #     * self.predict will need to be mocked to prevent the NotImplemented Exception

    # METHOD: add(self, X, y)
    # VALIDATE:
    #     * Test attribute values after
    # TODO:
    #     * Split this method in smaller ones
