from __future__ import division
import numpy as np
from scipy.stats import norm

from btb.tuning import Tuner, Uniform
from sklearn.gaussian_process import GaussianProcess, GaussianProcessRegressor


class GP(Tuner):
    def __init__(self, optimizables, **kwargs):
        """
        r_min: the minimum number of past results this selector needs in order
        to use gaussian process for prediction. If not enough results are
        present during a fit(), subsequent calls to propose() will revert to
        uniform selection.
        """
        super(GP, self).__init__(optimizables, **kwargs)
        self.r_min = kwargs.pop('r_min', 2)
        self.uniform = True

    def fit(self, X, y):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)
            y: np.ndarray of scores
        """
        super(GP, self).fit(X, y)

        if X.shape[0] < self.r_min:
            self.uniform = True
            return
        else:
            self.uniform = False

        # old gaussian process code
        #self.gp = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
        #                          nugget=np.finfo(np.double).eps * 1000)
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.gp.fit(X, y)

    def predict(self, X):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)

        returns:
            y: np.ndarray of predicted scores
        """
        # old gaussian process code
        #return self.gp.predict(X, eval_MSE=True)
        y, stdev = self.gp.predict(X, return_std=True)
        return np.array(zip(y, stdev))

    def acquire(self, predictions):
        """
        Predictions from the GP will be in the form (prediction, error).
        Default acquisition function just returns the highest prediction.
        """
        return np.argmax(predictions[0, :])

    def propose(self):
        """
        Using the probability_of_random value we computed in fit, either return
        the value with the best expected improvement or choose parameters
        randomly.
        """
        if self.uniform:
            # we probably don't have enough
            print 'GP: not enough data, falling back to uniform sampler'
            return Uniform(self.optimizables).propose()
        else:
            # otherwise do the normal generate-predict thing
            print 'GP: using gaussian process to select parameters'
            return super(GP, self).propose()


class GPEi(GP):
    def acquire(self, predictions):
        """
        Expected improvement criterion:
        http://people.seas.harvard.edu/~jsnoek/nips2013transfer.pdf
        """
        y_est, stdev = predictions.T
        best_y = max(self.y)

        # even though best_y is scalar and the others are vectors, this works
        z_score = (best_y - y_est) / stdev
        ei = stdev * (z_score * norm.cdf(z_score) + norm.pdf(z_score))

        return np.argmax(ei)


class GPEiVelocity(GPEi):
    MULTIPLIER = -100   # magic number; modify with care
    N_BEST_Y = 5        # number of top values w/w to compute velocity

    def fit(self, X, y):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)
            y: np.ndarray of scores
        """
        # first, train a gaussian process like normal
        super(GPEiVelocity, self).fit(X, y)

        self.probability_of_random = 0
        if len(y) >= self.r_min:
            # get the best few scores so far, and compute the average distance
            # between them.
            top_y = sorted(y)[-self.N_BEST_Y:]
            velocities = [top_y[i+1] - top_y[i] for i in range(len(top_y) - 1)]

            # the probability of returning random params scales inversely with
            # density of top scores.
            self.probability_of_random = np.exp(self.MULTIPLIER *
                                                np.mean(velocities))

    def propose(self):
        """
        Using the probability_of_random value we computed in fit, either return
        the value with the best expected improvement or choose parameters
        randomly.
        """
        if np.random.random() < self.probability_of_random:
            # choose params at random to avoid local minima
            return Uniform(self.optimizables).propose()
        else:
            # otherwise do the normal GPEi thing
            return super(GPEiVelocity, self).propose()
