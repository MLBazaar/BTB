# -*- coding: utf-8 -*-

"""Package where the boolean hyperparameters are defined."""

import numpy as np

from btb.tuning.hyperparams.base import BaseHyperParam


class BooleanHyperParam(BaseHyperParam):
    """Boolean Hyperparameter Class.

    The boolean hyperparameter class it is responsible for the transformation of boolean values in
    to normalized search space of [0, 1]^K and also providing samples of those or the inverse
    transformation from search space to hyperparameter space.

    Hyperparameter space:
        ``{True, False}``

    Search space:
        ``{0, 1}^K``

    Attributes:
        K (int):
            Number of dimensions that this hyperparameter uses to be represented in the search
            space.
    """

    K = 1

    def _within_hyperparam_space(self, values):
        if values.dtype is not np.dtype('bool'):
            raise ValueError('Values: {} not within hyperparameter space.'.format(values))

    def _inverse_transform(self, values):
        """Invert one or more values from search space [0, 1]^K.

        Converts a ``numpy.ndarray`` with normalized values from the search space [0, 1]^K to the
        original space of ``True`` and ``False`` by casting those values to ``boolean``.

        Args:
            values (numpy.ndarray):
                2D array of normalized values.

        Returns:
            denormalized (numpy.ndarray):
                2D array of denormalized values.

        Examples:
            >>> instance = BooleanHyperParam()
            >>> instance._inverse_transform(np.array([[0], [1]]))
            array([[True],
                   [False]])
            >>> instance._inverse_transform(np.array([[0]]))
            array([[False]])
        """
        return values.astype(bool)

    def _transform(self, values):
        """Transform one or more boolean values.

        Converts a ``numpy.array`` with values from the original hyperparameter space into
        normalized values in the search space of [0, 1]^K by casting those values to ``int``.

        Args:
            values (numpy.ndarray):
                2D array of values to be normalized.

        Returns:
            normalized (numpy.ndarray):
                2D array of shape(len(values), self.K).

        Examples:
            >>> instance = BooleanHyperParam()
            >>> instance._transform(np.array([[True], [False]]))
            array([[1],
                   [0]])
            >>> instance._transform(np.array([[True]])
            array([[1]])
        """
        return values.astype(int)

    def sample(self, n_samples):
        """Generate sample values in the hyperparameter search space of [0, 1]^k.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            samples (numpy.ndarray):
                2D array with shape of (n_samples, self.K) with normalized values inside the
                search space [0, 1]^k.

        Examples:
            >>> instance = BooleanHyperParam()
            >>> instance.sample(2)
            array([[1],
                   [1]])

        """
        sampled = np.random.random((n_samples, self.K))

        return np.round(sampled).astype(int)
