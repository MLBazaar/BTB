from btb.tuning import BaseTuner
import numpy as np


class Uniform(BaseTuner):
    """
    Very bare_bones tuner that returns a random set of parameters each time.
    """
    def predict(self, x):
        return np.random.rand(x.shape[0], 1)
