import logging
from builtins import range, object
import numpy as np
import random
import math

from btb import ParamTypes, EXP_TYPES

logger = logging.getLogger('btb')


class Tuner(object):
    def __init__(self, tunables, gridding=0, **kwargs):
        """
        Args:
            tunables: Ordered list of hyperparameter names and metadata
                objects. These describe the hyperparameters that this Tuner will
                be tuning. e.g.:
                [('degree', HyperParameter(type='INT', range=(2, 4))),
                 ('coef0', HyperParameter('INT', (0, 1))),
                 ('C', HyperParameter('FLOAT_EXP', (1e-05, 100000))),
                 ('gamma', HyperParameter('FLOAT_EXP', (1e-05, 100000)))]
            gridding: int. If a positive integer, controls the number of points
                on each axis of the grid. If 0, gridding does not occur.
        """
        self.tunables = tunables
        self.grid = gridding > 0

        if self.grid:
            self.grid_size = gridding
            self._define_grid()

    def _define_grid(self):
        """
        Define the range of possible values for each of the tunable
        hyperparameters.
        """
        self._grid_axes = []
        for _, param in self.tunables:
            if param.type == ParamTypes.INT:
                vals = np.round(np.linspace(param.range[0], param.range[1],
                                            self.grid_size))

            elif param.type == ParamTypes.FLOAT:
                vals = np.round(np.linspace(param.range[0], param.range[1],
                                            self.grid_size), decimals=5)

            # for exponential types, generate the grid in logarithm space so
            # that grid points will be expnentially distributed.
            elif param.type == ParamTypes.INT_EXP:
                vals = np.round(10.0 ** np.linspace(math.log10(param.range[0]),
                                                    math.log10(param.range[1]),
                                                    self.grid_size))

            elif param.type == ParamTypes.FLOAT_EXP:
                vals = np.round(10.0 ** np.linspace(math.log10(param.range[0]),
                                                    math.log10(param.range[1]),
                                                    self.grid_size), decimals=5)

            self._grid_axes.append(vals)

    def _params_to_grid(self, params):
        """
        Fit a vector of continuous parameters to the grid. Each parameter is
        fitted to the grid point it is closest to.
        """
        # This list will be filled with hyperparameter vectors that have been
        # mapped from vectors of continuous values to vectors of indices
        # representing points on the grid.
        grid_points = []
        for i, val in enumerate(params):
            axis = self._grid_axes[i]
            if self.tunables[i][1].type in EXP_TYPES:
                # if this is an exponential parameter, take the log of
                # everything before finding the closest grid point.
                # e.g. abs(4-1) < abs(4-10), but
                # abs(log(4)-log(1)) > abs(log(4)-log(10)).
                val = np.log(val)
                axis = np.log(axis)

            # find the index of the grid point closest to the hyperparameter
            # vector
            idx = min(range(len(axis)), key=lambda i: abs(axis[i] - val))
            grid_points.append(idx)

        return np.array(grid_points)

    def _grid_to_params(self, grid_points):
        """
        Map a single point on the grid, represented by indices into each axis,
        to a continuous-valued parameter vector.
        """
        params = [self._grid_axes[i][p] for i, p in enumerate(grid_points)]
        return np.array(params)

    def fit(self, X, y):
        """
        Args:
            X: np.array of hyperparameters,
                shape = (n_samples, len(tunables))
            y: np.array of scores, shape = (n_samples,)
        """
        self.X = X
        self.y = y

    def create_candidates(self, n=1000):
        """
        Generate a number of random hyperparameter vectors based on the
        specifications in self.tunables

        Args:
            n (optional): number of candidates to generate

        Returns:
            np.array of candidate hyperparameter vectors,
                shape = (n_samples, len(tunables))
        """
        # If using a grid, generate a list of previously unused grid points
        if self.grid:
            # convert numpy array to set of tuples of grid indices for easier
            # comparison
            past_vecs = set(tuple(self._params_to_grid(v)) for v in self.X)

            # if every point has been used before, gridding is done.
            num_points = self.grid_size ** len(self.tunables)
            if len(past_vecs) == num_points:
                return None

            # if fewer than n total points have yet to be seen, just return all
            # grid points not in past_vecs
            if num_points - len(past_vecs) <= n:
                # generate all possible points in the grid
                indices = np.indices(self._grid_axes)
                all_vecs = set(tuple(v) for v in
                               indices.T.reshape(-1, indices.shape[0]))
                vec_list = list(all_vecs - past_vecs)
            else:
                # generate n random vectors of grid-point indices
                vec_list = []
                for i in range(n):
                    # TODO: only choose from set of unused values
                    while True:
                        vec = np.random.randint(self.grid_size,
                                                size=len(self.tunables))
                        if tuple(vec) not in past_vecs:
                            break
                    vec_list.append(vec)

            # map the points back to continuous values and return
            return np.array([self._grid_to_params(v) for v in vec_list])

        # If not using a grid, generate a list of vectors where each parameter
        # is chosen uniformly at random
        else:
            # generate a matrix of random parameters, column by column.
            candidates = np.zeros((n, len(self.tunables)))
            for i, (k, param) in enumerate(self.tunables):
                lo, hi = param.range

                # TODO: move this to a HyperParameter class
                if param.type == ParamTypes.INT:
                    column = np.random.randint(lo, hi + 1, size=n)
                elif param.type == ParamTypes.FLOAT:
                    diff = hi - lo
                    column = lo + diff * np.random.rand(n)
                elif param.type == ParamTypes.INT_EXP:
                    column = 10.0 ** np.random.randint(math.log10(lo),
                                                       math.log10(hi) + 1,
                                                       size=n)
                elif param.type == ParamTypes.FLOAT_EXP:
                    diff = math.log10(hi) - math.log10(lo)
                    floats = math.log10(lo) + diff * np.random.rand(n)
                    column = 10.0 ** floats
                else:
                    logger.warn('Parameter passed with unknown type: %s' % param.type)

                candidates[:, i] = column
                i += 1

            return candidates

    def predict(self, X):
        """
        Args:
            X: np.array of hyperparameters,
                shape = (n_samples, len(tunables))

        Returns:
            y: np.array of predicted scores, shape = (n_samples)
        """
        raise NotImplementedError(
            'predict() needs to be implemented by a subclass of Tuner.')

    def acquire(self, predictions):
        """
        Acquisition function. Accepts a list of predicted values for candidate
        parameter sets, and returns the index of the best candidate.

        Args:
            predictions: np.array of predictions, corresponding to a set of
                proposed hyperparameter vectors. Each prediction may be a
                sequence with more than one value.

        Returns:
            idx: index of the selected hyperparameter vector
        """
        return np.argmax(predictions)

    def propose(self):
        """
        Use the trained model to propose a new set of parameters.

        Returns:
            proposal: np.array of proposed hyperparameter values, in the same
                order as self.tunables
        """
        # generate a list of random candidate vectors. If self.grid == True,
        # each candidate will be a vector that has not been used before.
        candidate_params = self.create_candidates()

        # create_candidates() returns None when every grid point has been tried
        if candidate_params is None:
            return None

        # predict() returns a tuple of predicted values for each candidate
        predictions = self.predict(candidate_params)

        # acquire() evaluates the list of predictions, selects one, and returns
        # its index.
        idx = self.acquire(predictions)
        return candidate_params[idx, :]
