import numpy as np
import random
import math

from btb import ParamTypes


class Tuner(object):
    def __init__(self, optimizables, grid=False, **kwargs):
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

        if grid:
            self.grid_size = kwargs.pop('grid_size', 3)
            self._define_grid()

    def _define_grid(self):
        """
        Define the range of possible values for each of the optimizable
        parameters.
        """
        self._grid_values = {}
        for i, struct in self.optimizables:
            if struct.type == ParamTypes.INT:
                vals = np.round(np.linspace(struct.range[0], struct.range[1],
                                            self.grid_size))

            elif struct.type == ParamTypes.INT_EXP:
                vals = np.round(10.0 ** np.linspace(math.log10(struct.range[0]),
                                                    math.log10(struct.range[1]),
                                                    self.grid_size))

            elif struct.type == ParamTypes.FLOAT:
                vals = np.round(np.linspace(struct.range[0], struct.range[1],
                                            self.grid_size), decimals=5)

            elif struct.type == ParamTypes.FLOAT_EXP:
                vals = np.round(10.0 ** np.linspace(math.log10(struct.range[0]),
                                                    math.log10(struct.range[1]),
                                                    self.grid_size), decimals=5)

            self._grid_values[i] = vals

    def _params_to_grid(self, params):
        """
        Fit a vector of continuous parameters to the grid. Each parameter is
        fitted to the grid point it is closest to.
        """
        grid_params = []
        exp_types = [ParamTypes.INT_EXP, ParamTypes.FLOAT_EXP]
        for val in params:
            grid = self._grid_values[i]
            if self.optimizables[i].type in exp_types:
                # if this is an exponential parameter, take the log of
                # everything before finding the closest grid point
                val = np.log(val)
                grid = np.log(grid)

            # find the index of the grid point closest to the continuous value
            idx = min(range(len(grid)), key=lambda i: abs(grid[i] - val))
            grid_params.append(idx)

        return np.array(grid_params)

    def _grid_to_params(self, grid_params):
        """
        Map indices of grid points to continuous values
        """
        params = [self._grid_values[i][p] for i, p in enumerate(grid_params)]
        return np.array(params)

    def fit(self, X, y):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)
            y: np.ndarray of scores
        """
        if self.grid:
            # If we're using gridding, map everything to indices on the grid
            self.X = np.array(map(self._params_to_grid, X))
        else:
            self.X = X
        self.y = y

    def create_candidates(self, n=1000):
        """
        Generate a number of random hyperparameter vectors based on the
        parameter specifications given to the constructor.
        """
        if self.grid:
            past_params = set(tuple(v) for v in self.X)

            # if every point has been used before, gridding is done.
            total_points = self.grid_size ** len(self.optimizables)
            if len(past_params) == total_points:
                return None

            vec_list = []
            n_candidates = min(total_points - len(past_params), n)

            # generate up to n random vectors of grid-point indices
            for i in xrange(n_candidates):
                # TODO: only choose from set of unused values
                while True:
                    vec = np.random.randint(self.grid_size,
                                            size=len(self.optimizables))
                    if vec not in past_params:
                        break
                vec_list.append(vec)
            vectors = np.array(vec_list)

        else:
            vectors = np.zeros((n, len(self.optimizables)))
            for i, (k, struct) in enumerate(self.optimizables):
                lo, hi = struct.range

                if struct.type == ParamTypes.INT:
                    column = np.random.randint(lo, hi + 1, size=n)
                elif struct.type == ParamTypes.INT_EXP:
                    column = 10.0 ** np.random.randint(math.log10(lo),
                                                       math.log10(hi) + 1,
                                                       size=n)
                elif struct.type == ParamTypes.FLOAT:
                    diff = hi - lo
                    column = lo + diff * np.random.rand(n)
                elif struct.type == ParamTypes.FLOAT_EXP:
                    diff = math.log10(hi) - math.log10(lo)
                    floats = math.log10(lo) + diff * np.random.rand(n)
                    column = 10.0 ** floats

                vectors[:, i] = column
                i += 1

            return vectors

    def predict(self, X):
        """
        Args:
            X: np.ndarray of feature vectors (vectorized parameters)

        returns:
            y: np.ndarray of predicted scores
        """
        raise NotImplementedError(
            'predict() needs to be implemented by a subclass of Tuner.')

    def acquire(self, predictions):
        """
        Acquisition function. Accepts a list of predicted values for candidate
        parameter sets, and returns the index of the best candidate.
        """
        return np.argmax(predictions)

    def propose(self):
        """
        Use the trained model to propose a new set of parameters.
        """
        candidate_params = self.create_candidates()
        if candidate_params is None:
            return None

        predictions = self.predict(candidate_params)
        best = self.acquire(predictions)

        if self.grid:
            return self._grid_to_params(candidate_params[best, :])
        else:
            return candidate_params[best, :]
