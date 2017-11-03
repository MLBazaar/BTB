import numpy as np
import random
import math

from btb import ParamTypes


class Tuner(object):
    def __init__(self, optimizables, **kwargs):
        """
        Accepts a list of pamameter metadata structures.
        optimizables will look like this:
        [
            ('degree', HyperParameter(range=(2, 4),
                                 type='INT',
                                 is_categorical=False)),
            ('coef0', HyperParameter((0, 1), 'INT', False)),
            ('C', HyperParameter((1e-05, 100000), 'FLOAT_EXP', False)),
            ('gamma', HyperParameter((1e-05, 100000), 'FLOAT_EXP', False))
        ]
        """
        self.optimizables = optimizables

    def fit(self, X, y):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)
            y: np.ndarray of scores
        """
        pass

    def predict(self, X):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)

        returns:
            y: np.ndarray of predicted scores
        """
        pass

    def create_candidates(self, n=1000):
        """
        Generate a number of random hyperparameter vectors based on the
        parameter specifications given to the constructor.
        """
        vectors = np.zeros((n, len(self.optimizables)))
        for i, (k, struct) in enumerate(self.optimizables):
            if struct.type == ParamTypes.FLOAT_EXP:
                random_powers = 10.0 ** np.random.random_integers(
                    math.log10(struct.range[0]), math.log10(struct.range[1]), size=n)
                random_floats = np.random.rand(n)
                column = np.multiply(random_powers, random_floats)

            elif struct.type == ParamTypes.INT:
                column = np.random.random_integers(struct.range[0],
                                                   struct.range[1], size=n)

            elif struct.type == ParamTypes.INT_EXP:
                column = 10.0 ** np.random.random_integers(
                    math.log10(struct.range[0]), math.log10(struct.range[1]),
                    size=n)

            elif struct.type == ParamTypes.FLOAT:
                column = np.random.rand(n)

            vectors[:, i] = column
            i += 1

        return vectors

    def propose(self):
        """
        Use the trained model to propose a new set of parameters.
        """
        candidate_params = self.create_candidates()
        predictions = self.predict(candidate_params)
        best = np.argmax(predictions)
        return candidate_params[best, :]
