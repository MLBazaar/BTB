from unittest import TestCase

# from btb.tuning.gp import GP, GPEi, GPEiVelocity


class TestGP(TestCase):

    # METHOD __init__(self, tunables, gridding=0, **kwargs)
    # VALIDATE:
    #     * attribute values
    # TODO:
    #     * Use an explicit parameter name instead of popping it from kwargs

    # METHOD: fit(self, X, y)
    # VALIDATE:
    #     * if X shorter than r_minimum, nothing is done
    #     * GaussianProcessRegressor is called with the right values.
    # NOTES:
    #     * GPR will need to be mocked.

    # METHOD: predict(self, X)
    # VALIDATE:
    #     * if X shorter than r_minimum, Uniform is used.
    #     * GPR is colled with the right values
    # NOTES:
    #     * GPR will need to be mocked

    # METHOD: _acquire(self, predictions)
    # VALIDATE:
    #     * np.argmax is called with the right values
    # NOTES:
    #     * np.argmax will need to be mocked

    pass


class TestGPEi(TestCase):

    # METHOD: _acquire(self, predictions)
    # VALIDATE:
    #     * return values according to the formula

    pass


class TestGPEiVelocity(TestCase):

    # METHOD: fit(self, X, y)
    # VALIDATE:
    #     * if y shorter than r_minimum, nothing is done
    #     * POU attribute values according to the formula

    # METHOD: predict(self, X, y)
    # VALIDATE:
    #     * cases where Uniform is returned
    # NOTES:
    #     * random will need to be mocked
    pass
