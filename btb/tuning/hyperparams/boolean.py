# -*- coding: utf-8 -*-

"""Package where the BooleanHyperParam class is defined."""

import numpy as np

from btb.tuning.hyperparams.base import BaseHyperParam


class BooleanHyperParam(BaseHyperParam):
    """BooleanHyperParam class.

    The BooleanHyperParam class is responsible for the transformation of boolean values in
    to normalized search space of :math:`[0, 1]`, providing the ability to sample values of those
    and to inverse transform from the search space into the hyperparameter space.

    Hyperparameter space:
        ``{True, False}``

    Args:
        default (bool):
            Default boolean value for the hyperparameter. Defaults to ``False``.
    """

    dimensions = 1
    cardinality = 2

    def __init__(self, default=False):
        self.default = default

    def _within_hyperparam_space(self, values):
        if values.dtype is not np.dtype('bool'):
            # values is expected to be np.ndarray(n, 1) [[False], [True]]
            if not all(isinstance(value[0], bool) for value in values):
                raise ValueError('Values: {} not within hyperparameter space.'.format(values))

    def _inverse_transform(self, values):
        """Invert one or more search space values.

        Converts a ``numpy.ndarray`` with normalized values from the search space :math:`{0, 1}`
        to the original space of ``True`` and ``False`` by casting those values to ``boolean``.

        Args:
            values (numpy.ndarray):
                2D array with values from the search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` containing values from the original hyperparameter space.

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
        r"""Transform one or more hyperparameter values.

        Converts a ``numpy.array`` with values from the original hyperparameter space into
        normalized values in the search space of :math:`\{0, 1\}` by casting those values to
        ``int``.

        Args:
            values (numpy.ndarray):
                2D array with values from the hyperparameter space to be converted into the
                search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` of shape ``(len(values), 1)`` containing the search space
                values.

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
        """Generate sample values in the hyperparameter search space :math:`{0, 1}`.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            numpy.ndarray:
                2D array with shape of ``(n_samples, 1)`` with normalized values inside the
                search space :math:`{0, 1}`.

        Example:
            The example below shows simple usage case where a BooleanHyperParam is being created
            and it's ``sample`` method is being called with a number of samples to be obtained.

            >>> instance = BooleanHyperParam()
            >>> instance.sample(2)
            array([[1],
                   [1]])
        """
        sampled = np.random.random((n_samples, self.dimensions))

        return np.round(sampled).astype(int)

    def __repr__(self):
        return 'BooleanHyperParam(default={})'.format(self.default)
