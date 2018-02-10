from __future__ import division
import logging
from builtins import zip, range

import numpy as np
from scipy.stats import norm

from btb.tuning import Tuner, Uniform
from sklearn.gaussian_process import GaussianProcess, GaussianProcessRegressor

logger = logging.getLogger('btb')


class GP(Tuner):
    def __init__(self, tunables, gridding=0, **kwargs):
        """
        Extra args:
            r_minimum: the minimum number of past results this selector needs in
                order to use gaussian process for prediction. If not enough
                results are present during a fit(), subsequent calls to
                propose() will revert to uniform selection.
        """
        super(GP, self).__init__(tunables, gridding=gridding, **kwargs)
        self.r_minimum = kwargs.pop('r_minimum', 2)

    def fit(self, X, y):
        """ Use X and y to train a Gaussian process. """
        super(GP, self).fit(X, y)

        # skip training the process if there aren't enough samples
        if X.shape[0] < self.r_minimum:
            return

        # old gaussian process code
        #self.gp = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
        #                          nugget=np.finfo(np.double).eps * 1000)
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.gp.fit(X, y)

    def predict(self, X):
        # old gaussian process code
        #return self.gp.predict(X, eval_MSE=True)
        y, stdev = self.gp.predict(X, return_std=True)
        return np.array(list(zip(y, stdev)))

    def acquire(self, predictions):
        """
        Predictions from the GP will be in the form (prediction, error).
        The default acquisition function returns the index with the highest
        predicted value, not factoring in error.
        """
        return np.argmax(predictions[:, 0])

    def propose(self):
        """
        If we haven't seen at least self.r_minimum values, choose parameters
        using a Uniform tuner (randomly). Otherwise perform the usual
        create-predict-propose pipeline.
        """
        if self.X.shape[0] < self.r_minimum:
            # we probably don't have enough
            logger.warn('GP: not enough data, falling back to uniform sampler')
            return Uniform(self.tunables).propose()
        else:
            # otherwise do the normal generate-predict thing
            logger.info('GP: using gaussian process to select parameters')
            return super(GP, self).propose()


class GPEi(GP):
    def acquire(self, predictions):
        """
        Expected improvement criterion:
        http://people.seas.harvard.edu/~jsnoek/nips2013transfer.pdf
        Args:
            predictions: np.array of (estimated y, estimated error) tuples that
                the gaussian process generated for a series of
                proposed hyperparameters.
        """
        y_est, stderr = predictions.T
        best_y = max(self.y)

        # even though best_y is scalar and the others are vectors, this works
        z_score = (best_y - y_est) / stderr
        ei = stderr * (z_score * norm.cdf(z_score) + norm.pdf(z_score))

        return np.argmax(ei)


class GPEiVelocity(GPEi):
    MULTIPLIER = -100   # magic number; modify with care
    N_BEST_Y = 5        # number of top values w/w to compute velocity

    def fit(self, X, y):
        """
        Train a gaussian process like normal, then compute a "Probability Of
        Uniform selection" (POU) value.
        """
        # first, train a gaussian process like normal
        super(GPEiVelocity, self).fit(X, y)

        # probability of uniform
        self.POU = 0
        if len(y) >= self.r_minimum:
            # get the best few scores so far, and compute the average distance
            # between them.
            top_y = sorted(y)[-self.N_BEST_Y:]
            velocities = [top_y[i+1] - top_y[i] for i in range(len(top_y) - 1)]

            # the probability of returning random parameters scales inversely with
            # the "velocity" of top scores.
            self.POU = np.exp(self.MULTIPLIER * np.mean(velocities))

    def propose(self):
        """
        Use the POU value we computed in fit to choose randomly between GPEi and
        uniform random selection.
        """
        if np.random.random() < self.POU:
            # choose params at random to avoid local minima
            return Uniform(self.tunables).propose()
        else:
            # otherwise do the normal GPEi thing
            return super(GPEiVelocity, self).propose()
