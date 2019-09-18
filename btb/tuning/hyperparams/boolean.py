# -*- coding: utf-8 -*-

"""Package where the boolean hyperparameters are defined."""

import numpy as np

from btb.tuning.hyperparams.base import BaseHyperParam


class BooleanHyperParam(BaseHyperParam):
    """Boolean Hyperparameter Class.

    The boolean hyperparameter class it is responsible for the transformation of boolean values in
    to normalized search space of [0, 1] and also providing samples of those or the inverse
    transformation from search space to hyperparameter space.

    Hyperparameter space:
        ``{True, False}``
    """

    K = 1

    def _within_hyperparam_space(self, values):
        if values.dtype is not np.dtype('bool'):
            raise ValueError('Values: {} not within hyperparameter space.'.format(values))

    def _inverse_transform(self, values):
        """Invert one or more values from search space {0, 1}.

        Converts a ``numpy.ndarray`` with normalized values from the search space {0, 1} to the
        original space of ``True`` and ``False`` by casting those values to ``boolean``.

        Args:
            values (numpy.ndarray):
                2D array of normalized values.

        Returns:
            numpy.ndarray:
                2D array of denormalized values.

        Example:
            The example below shows simple usage case where a BooleanHyperParam is being created
            and it's ``_inverse_transform`` method is being called with a valid ``numpy.ndarray``
            containing a valid shape of data.

            >>> instance = BooleanHyperParam()
            >>> instance._inverse_transform(np.array([[0], [1]]))
            array([[ True],
                   [False]])
            >>> instance._inverse_transform(np.array([[0]]))
            array([[False]])
        """
        return values.astype(bool)

    def _transform(self, values):
        """Transform one or more boolean values.

        Converts a ``numpy.array`` with values from the original hyperparameter space into
        normalized values in the search space of {0, 1} by casting those values to ``int``.

        Args:
            values (numpy.ndarray):
                2D array of values to be normalized.

        Returns:
            numpy.ndarray:
                2D array of shape(len(values), self.K).

        Example:
            The example below shows simple usage case where a BooleanHyperParam is being created
            and it's ``_transform`` method is being called with a valid ``numpy.ndarray``.

            >>> instance = BooleanHyperParam()
            >>> instance._transform(np.array([[True], [False]]))
            array([[1],
                   [0]])
            >>> instance._transform(np.array([[True]]))
            array([[1]])
        """
        return values.astype(int)

    def sample(self, n_samples):
        """Generate sample values in the hyperparameter search space of {0, 1}.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            numpy.ndarray:
                2D array with shape of (n_samples, self.K) with normalized values inside the
                search space {0, 1}.

        Example:
            The example below shows simple usage case where a BooleanHyperParam is being created
            and it's ``sample`` method is being called with a number of samples to be obtained.

            >>> instance = BooleanHyperParam()
            >>> instance.sample(2)
            array([[1],
                   [1]])
        """
        sampled = np.random.random((n_samples, self.K))

        return np.round(sampled).astype(int)
