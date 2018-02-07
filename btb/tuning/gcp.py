from __future__ import division
import logging
from builtins import zip, range

import numpy as np
import scipy.stats as st
from scipy.stats import norm

from btb.tuning import Tuner, Uniform
from sklearn.gaussian_process import GaussianProcess, GaussianProcessRegressor

logger = logging.getLogger('btb')


def make_cdf(kernel_pdf):
    from scipy.interpolate import interp1d

    def np_cdf(np_model, y):
        u = []
        for yi in y:
            ui = np_model.integrate_box_1d(-np.inf, yi)
            u.append(ui)
        u = np.asarray(u)
        return u

    lowerB = kernel_pdf.dataset.min()
    upperB = kernel_pdf.dataset.max()
    mid = (upperB-lowerB) / 2
    Nmid = -0.2
    lowerB = lowerB - mid * Nmid
    upperB = upperB + mid * Nmid
    num = int(10 ** (np.log10(upperB - lowerB)) // 1)
    if num < 100: num = 100
    y_range = np.linspace(lowerB, upperB, num=num)
    u_range = np_cdf(kernel_pdf, y_range)
    funI = interp1d(y_range, u_range)  # <-- use linear interpolation

    def kernel_cdf(y):
        u = []
        for yi in y:
            if yi < lowerB:
                ui = kernel_pdf.integrate_box_1d(-np.inf, yi)
            elif yi > upperB:
                ui = kernel_pdf.integrate_box_1d(-np.inf, yi)
            else:
                ui = funI(yi)
            u.append(ui)
        u = np.asarray(u)
        return u

    return kernel_cdf

def make_ppf(kernel_pdf):
    from scipy.interpolate import interp1d
    from scipy.optimize import fsolve

    lowerB = kernel_pdf.dataset.min()
    upperB = kernel_pdf.dataset.max()
    mid = (upperB-lowerB) / 2
    Nmid = -0.2
    lowerB = lowerB - mid * Nmid
    upperB = upperB + mid * Nmid
    num = int(10 ** (np.log10(upperB - lowerB)) // 1)
    if num < 100: num = 100
    y_range = np.linspace(lowerB, upperB, num=num)
    kernel_cdf = make_cdf(kernel_pdf)
    u_range = kernel_cdf(y_range)
    funI = interp1d(u_range, y_range)  # <-- use linear interpolation
    lowerU = min(u_range)
    upperU = max(u_range)

    def invCDF_eq(y, u):
        out = u - kernel_pdf.integrate_box_1d(-np.inf, y)  # <-- use fsolve
        return out

    def kernel_ppf(u):
        y = np.array([])
        for ui in u:
            if ui <= lowerU:
                yi = fsolve(invCDF_eq, lowerB, args=ui)
            elif ui >= upperU:
                yi = fsolve(invCDF_eq, upperB, args=ui)
            else:
                yi = funI(ui)
            y = np.hstack((y, yi))
        return y

    return kernel_ppf

class GCP(Tuner):
    def __init__(self, tunables, gridding=0, **kwargs):
        """
        Extra args:
            r_minimum: the minimum number of past results this selector needs in
                order to use gaussian process for prediction. If not enough
                results are present during a fit(), subsequent calls to
                propose() will revert to uniform selection.
        """
        super(GCP, self).__init__(tunables, gridding=gridding, **kwargs)
        self.r_minimum = kwargs.pop('r_minimum', 2)

    def fit(self, X, y):

        def jitter(x, range):
            y = np.copy(x)
            scale_exp_min = np.abs(np.ceil(np.log10(range[0])))
            scale_exp_max = np.abs(np.ceil(np.log10(range[1])))
            scale_exp = (scale_exp_max + scale_exp_min) /2.
            r = np.random.rand(y.size) / (10**scale_exp)
            y = y + r
            return y

        # Print msg. when going into gcp.fit
        strMessage = "rows in X = %d, r_minimum = %d" % (X.shape[0], self.r_minimum)
        logger.debug(strMessage)

        # Use X and y to train a Gaussian Copula Process.
        super(GCP, self).fit(X, y)

        # skip training the process if there aren't enough samples
        if X.shape[0] < self.r_minimum:
            return

        # -- Non-parametric model of 'y', estimated with kernel density
        kernel_pdf = st.gaussian_kde(y)
        kernel_cdf = make_cdf(kernel_pdf)
        kernel_ppf = make_ppf(kernel_pdf)
        y_kernel_model = {'pdf': kernel_pdf, 'cdf': kernel_cdf, 'ppf': kernel_ppf}
        self.y_kernel_model = y_kernel_model

        # - Transform y-->F-->vF-->norm.ppf-->v
        vF = y_kernel_model['cdf'](y)
        v = st.norm.ppf(vF)

        # -- Non-parametric model of each feature in 'X', estimated with kernel density
        X_kernel_model=[]
        for ki in range(X.shape[1]):
            columnX = X[:,ki]
            if self.tunables[ki][1].type != 'float':
                columnX = jitter(columnX,self.tunables[ki][1].range)
            kernel_pdf = st.gaussian_kde(columnX)
            kernel_cdf = make_cdf(kernel_pdf)
            kernel_ppf = make_ppf(kernel_pdf)
            kernel_model = {'pdf': kernel_pdf, 'cdf': kernel_cdf, 'ppf': kernel_ppf}
            X_kernel_model.append(kernel_model)
        self.X_kernel_model = X_kernel_model

        # -- Transform X-->F-->uF-->norm.ppf-->U
        U = np.empty_like(X)
        for ki in range(X.shape[1]):
            uF = X_kernel_model[ki]['cdf'](X[:,ki])
            U[:,ki] = st.norm.ppf(uF)

        # - Instantiate a GP and fit it with (U,v)
        self.gcp = GaussianProcessRegressor(normalize_y=True)
        self.gcp.fit(U, v)

    def predict(self, X):

        def get_valid_row(U):
            ind_OK = np.full(U.shape[0],1,dtype=bool)
            for ki in range(U.shape[1]):
                ind_OK = np.logical_and(ind_OK, np.logical_not(np.isinf(U[:,ki])) )
            V = np.copy(U[ind_OK,:])
            return V, ind_OK

        # -- Load non-parametric model
        x_kernel_model = self.X_kernel_model
        y_kernel_model = self.y_kernel_model

        # -- Transform X into U before using the GP learned
        U = np.empty_like(X)
        for ki in range(X.shape[1]):
            uF = x_kernel_model[ki]['cdf'](X[:,ki])
            U[:,ki] = st.norm.ppf(uF)

        #-- Get U_safe and print msg. to inform of how many rows are valid
        U_safe,ind_OK = get_valid_row(U)
        strMessage = "Num. of valid rows in X = %d" % (np.sum(ind_OK))
        logger.debug(strMessage)

        # -- use GP to estimate mean and stdev only of safe U's
        mu_v, stdev_v = self.gcp.predict(U_safe, return_std=True)

        # -- Transform back mu_u-->NormStd-->mu_uF
        mu_vF = st.norm.cdf(mu_v)
        stdev_vF = st.norm.cdf(stdev_v)

        # -- Transform back mu_uF-->F.ppf-->mu_y
        # VERSION 1:
        # It should be used in case of mu_y and stdev_y can have a size lower than X.
        # Otherwise, swap to version 2.
        mu_y = y_kernel_model['ppf'](mu_vF)
        stdev_y = y_kernel_model['ppf'](stdev_vF)
        '''
        # VERSION 2:
        # It should be used in case of mu_y and stdev_y must have the same length
        # than X. Otherwise, Version 1 is faster.
        # -- Transform back mu_uF-->F.ppf-->mu_y
        #    mu_y has the same length than U, but is positive only for safe rows
        mu_y = np.zeros([U.shape[0]])
        stdev_y = np.zeros([U.shape[0]])
        mu_y[ind_OK] = y_kernel_model['ppf'](mu_vF)
        stdev_y[ind_OK] = y_kernel_model['ppf'](stdev_vF)
        '''

        return np.array(list(zip(mu_y, stdev_y)))

    def acquire(self, predictions):
        """
        Predictions from the GCP will be in the form (prediction, error).
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
            logger.info('GCP: using gaussian copula process to select parameters')
            return super(GCP, self).propose()


class GCPEi(GCP):
    #-- question: I have changed GPEi(GP) for GPEi(GCP), is that ok?
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


class GCPEiVelocity(GCPEi):
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
