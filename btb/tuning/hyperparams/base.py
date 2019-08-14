# -*- coding: utf-8 -*-

"""Package where the base hyperparameter is defined."""

from abc import ABCMeta, abstractmethod

import numpy as np


def to_array(values, dimension):
    """Perform a validation if the data can be converted to array for the hyperparameter.

    Args:
        values(scalar, ArrayLike):
            A scalar value or ArrayLike values to be converted to ``numpy.array``.
        dimension(int):
            The dimension that the hyperparameter has.

    Returns:
        values(Array):
            Values converted in to ``array`` of shape ``(n, dimension)`` where ``n`` is the
            length of values.

    Raises:
        ValueError:
            A ``ValueError`` is raised if any value from ``values`` is not within the dimension.
    """

    if not isinstance(values, (list, np.ndarray)):
        if dimension != 1:
            raise ValueError('Value not in dimension.')

        values = [[values]]

    else:
        if dimension != 1:
            if not isinstance(values[0], (list, np.ndarray)):
                values = [values]

            if not all(len(value) == dimension for value in values):
                raise ValueError('Values not in dimension.')
        else:
            if not isinstance(values[0], (list, np.ndarray)):
                values = [[value] for value in values]

    return np.array(values)


class BaseHyperParam(metaclass=ABCMeta):
    """Base hyperparameter class.

    Abstract representation of a single hyperparameter that needs to be tuned.

    Attributes:
        K (int):
            Number of dimensions that this HyperParam uses to be represented in the search space.
    """

    def _within_range(self, values, min=0, max=1):
        """Ensure that the values are between a certain range.

        Raises:
            ValueError:
                A ``ValueError`` is raised if any value from ``values`` is not inside the range.
        """
        inside_mask = np.ma.masked_inside(values, min, max)
        if inside_mask.any():
            outside = inside_mask[~inside_mask.mask].data.reshape(-1).tolist()
            raise ValueError(
                'Values found outside of the valid range [{}, {}]: {}'.format(min, max, outside)
            )

    def _within_hyperparam_space(self, values):
        self._within_range(values, self.min, self.max)

    def _within_search_space(self, values):
        self._within_range(values, 0, 1)

    @abstractmethod
    def _inverse_transform(self, values):
        pass

    @abstractmethod
    def _transform(self, values):
        pass

    def inverse_transform(self, values):
        """Invert one or more hyperparameter values.

        Transform one or more hyperparameter values from the normalized search
        space [0, 1]^k to the original hyperparameter space.

        Args:
            values (ArrayLike):
                Single value or 2D ArrayLike of normalized values.

        Returns:
            denormalized (numpy.ndarray):
                2D array of denormalized values.
        """
        values = to_array(values, self.K)
        self._within_search_space(values)

        return self._inverse_transform(values)

    @abstractmethod
    def sample(self, n_samples):
        """Generate an array of ``n_samples`` random samples in the search space.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            samples (numpy.ndarray):
                2D array with of shape (n_samples, self.K)
        """

    def transform(self, values):
        """Transform one or more hyperparameter values.

        Transform one or more hyperparameter values from the original hyperparameter space to the
        normalized search space [0, 1]^k.
        The accepted value formats are:

            - Single value:
                A single value from the original hyperparameter space.
            - List:
                A list composed by values from the original hyperparameter space.
            - 2D Array:
                Two dimension array like object that contains values from the original
                hyperparameter space.

        Args:
            values (Union[object, List[object]]):
                Single value or list of values to normalize.

        Returns:
            normalized (numpy.ndarray):
                2D array of shape(len(values), self.K)

        Examples:
            >>> from btb.tuning.hyperparams.numerical import IntHyperParam
            >>> ihp = IntHyperParam(min=1, max=4)
            >>> ihp.transform(1)
            array([[0.125]])

            >>> ihp.transform([1, 2])
            array([[0.125],
                   [0.375]])

            >>> ihp.transform([[1], [2]])
            array([[0.125],
                   [0.375]])
        """

        if not isinstance(values, np.ndarray):
            values = np.asarray(values)

        dimensions = len(values.shape)
        if dimensions > 2:
            raise ValueError('Too many dimensions.')

        elif dimensions < 2:
            values = values.reshape(-1, 1)

        if values.shape[1] > 1:
            raise ValueError('Only one column is supported.')

        self._within_hyperparam_space(values)

        return self._transform(values)
