import itertools
import logging

import numpy as np

logger = logging.getLogger('btb')


class BaseTuner(object):
    """Base tuner

    Args:
        tunables (List[Tuple[str, HyperParameter]]): Ordered list of hyperparameter names and
            metadata objects. These describe the hyperparameters that this Tuner will be tuning,
            e.g.::

                [
                    ('degree', HyperParameter(type='INT', range=(2, 4))),
                    ('coef0', HyperParameter('INT', (0, 1))),
                    ('C', HyperParameter('FLOAT_EXP', (1e-05, 100000))),
                    ('gamma', HyperParameter('FLOAT_EXP', (1e-05, 100000)))
                ]

        gridding (int): If a positive integer, controls the number of points on each axis of the
            grid. If 0, gridding does not occur.
    """

    def __init__(self, tunables, gridding=0):
        self.tunables = tunables
        self.grid = gridding > 0
        self._best_score = -1 * float('inf')
        self._best_hyperparams = None

        if self.grid:
            self.grid_width = gridding
            self._grid_axes = self._generate_grid()

        self.X_raw = None
        self.y_raw = []

        self.X = np.array([])
        self.y = np.array([])

    def _generate_grid(self):
        """Get the all possible values for each of the tunables."""
        grid_axes = []
        for _, param in self.tunables:
            grid_axes.append(param.get_grid_axis(self.grid_width))

        return grid_axes

    def fit(self, X, y):
        """Fit

        Args:
            X (np.array): Array of hyperparameter values with shape (n_samples, len(tunables))
            y (np.array): Array of scores with shape (n_samples, )
        """
        self.X = X
        self.y = y

    def _candidates_from_grid(self, n=1000):
        """Get unused candidates from the grid or parameters."""
        used_vectors = set(tuple(v) for v in self.X)

        # if every point has been used before, gridding is done.
        grid_size = self.grid_width ** len(self.tunables)
        if len(used_vectors) == grid_size:
            return None

        all_vectors = set(itertools.product(*self._grid_axes))
        remaining_vectors = all_vectors - used_vectors
        candidates = np.array(list(map(np.array, remaining_vectors)))

        np.random.shuffle(candidates)
        return candidates[0:n]

    def _random_candidates(self, n=1000):
        """Generate a matrix of random parameters, column by column."""

        candidates = np.zeros((n, len(self.tunables)))
        for i, tunable in enumerate(self.tunables):
            param = tunable[1]
            lo, hi = param.range
            if param.is_integer:
                column = np.random.randint(lo, hi + 1, size=n)

            else:
                diff = hi - lo
                column = lo + diff * np.random.rand(n)

            candidates[:, i] = column

        return candidates

    def _create_candidates(self, n=1000):
        """Generate random hyperparameter vectors

        Args:
            n (int, optional): number of candidates to generate. Defaults to 1000.

        Returns:
            candidates (np.array): Array of candidate hyperparameter vectors with shape
                (n_samples, len(tunables))
        """
        # If using a grid, generate a list of previously unused grid points
        if self.grid:
            return self._candidates_from_grid(n)

        # If not using a grid, generate a list of vectors where each parameter
        # is chosen uniformly at random
        else:
            return self._random_candidates(n)

    def predict(self, X):
        """Predict
        Args:
            X (np.array): Array of hyperparameters with shape  (n_samples, len(tunables))

        Returns:
            np.array: Array of predicted scores with shape (n_samples)
        """
        raise NotImplementedError

    def _acquire(self, predictions):
        """Acquire

        Acquisition function. Accepts a list of predicted values for candidate parameter sets, and
            returns the index of the best candidate.

        Args:
            predictions (np.array): Array of predictions, corresponding to a set of proposed
                hyperparameter vectors. Each prediction may be a sequence with more than one value.

        Returns:
            int: index of the selected hyperparameter vector
        """
        return np.argmax(predictions)

    def propose(self, n=1):
        """Use the trained model to propose a new set of parameters.

        Args:
            n (int, optional): number of candidates to propose

        Returns:
            Mapping of tunable name to proposed value. If called with n>1 then proposal is a list
                of dictionaries.
        """
        proposed_params = []

        for i in range(n):
            # generate a list of random candidate vectors. If self.grid == True
            # each candidate will be a vector that has not been used before.
            candidate_params = self._create_candidates()

            # create_candidates() returns None when every grid point
            # has been tried
            if candidate_params is None:
                return None

            # predict() returns a tuple of predicted values for each candidate
            predictions = self.predict(candidate_params)

            # acquire() evaluates the list of predictions, selects one,
            # and returns its index.
            idx = self._acquire(predictions)

            # inverse transform acquired hyperparameters
            # based on hyparameter type
            params = {}
            for i in range(candidate_params[idx, :].shape[0]):
                inverse_transformed = self.tunables[i][1].inverse_transform(
                    candidate_params[idx, i]
                )
                params[self.tunables[i][0]] = inverse_transformed
            proposed_params.append(params)

        return params if n == 1 else proposed_params

    def add(self, X, y):
        """Add data about known tunable hyperparameter configurations and scores.

        Refits model with all data.

        Args:
            X (Union[Dict[str, object], List[Dict[str, object]]]): dict or list of dicts of
                hyperparameter combinations. Keys may only be the name of a tunable, and the
                dictionary must contain values for all tunables.
            y (Union[float, List[float]]): float or list of floats of scores of the hyperparameter
                combinations. Order of scores must match the order of the hyperparameter
                dictionaries that the scores corresponds
        """
        if isinstance(X, dict):
            X = [X]
            y = [y]

        # transform the list of dictionaries into a np array X_raw
        for i in range(len(X)):
            each = X[i]
            # update best score and hyperparameters
            if y[i] > self._best_score:
                self._best_score = y[i]
                self._best_hyperparams = X[i]

            vectorized = []
            for tunable in self.tunables:
                vectorized.append(each[tunable[0]])

            if self.X_raw is not None:
                self.X_raw = np.append(
                    self.X_raw,
                    np.array([vectorized], dtype=object),
                    axis=0,
                )

            else:
                self.X_raw = np.array([vectorized], dtype=object)

        self.y_raw = np.append(self.y_raw, y)

        # transforms each hyperparameter based on hyperparameter type
        x_transformed = np.array([], dtype=np.float64)
        if len(self.X_raw.shape) > 1 and self.X_raw.shape[1] > 0:
            x_transformed = self.tunables[0][1].fit_transform(
                self.X_raw[:, 0],
                self.y_raw,
            ).astype(float)

            for i in range(1, self.X_raw.shape[1]):
                transformed = self.tunables[i][1].fit_transform(
                    self.X_raw[:, i],
                    self.y_raw,
                ).astype(float)
                x_transformed = np.column_stack((x_transformed, transformed))

        self.fit(x_transformed, self.y_raw)
