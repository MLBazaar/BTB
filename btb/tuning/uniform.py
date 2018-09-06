import numpy as np

from btb.tuning.tuner import BaseTuner


class Uniform(BaseTuner):
    """Uniform tuner

    Selects a new hyperparameter configuration uniformly at random
    """

    def predict(self, x):
        return np.random.rand(x.shape[0], 1)
