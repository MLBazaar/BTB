from btb.tuning import Tuner
import numpy as np

class Uniform(Tuner):
    """
    Very bare_bones tuner that returns a random set of parameters each time.
    """
    def predict(self, x):
        return np.random.rand(x.shape[0], 1)
