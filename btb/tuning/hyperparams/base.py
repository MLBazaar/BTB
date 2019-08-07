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
            Values converted in to ``array`` of shape ``(n, dimension)`` where ``np`` is the
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

    values = np.array(values)

    return values


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
        if (values < min).any() or (values > max).any():
            raise ValueError('Value not within range.')

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
        """Revert one or more hyperparameter values.

        Transform one or more hyperparameter values from the normalized search
        space [0, 1]^k to the original hyperparameter space.

        Args:
            values (ArrayLike):
                Single value or 2D ArrayLike of normalized values.

        Returns:
            denormalized (Union[object, List[object]]):
                Denormalized value or list of denormalized values.
        """
        values = to_array(values, self.K)
        self._within_search_space(values)

        return self._inverse_transform(values)

    @abstractmethod
    def sample(self, n_samples):
        """Generate an array of ``num_samples`` random samples in the search space.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            samples (ArrayLike):
                2D array with of shape (n_samples, self.K)
        """
        pass

    def transform(self, values):
        """Transform one or more hyperparameter values.

        Transform one or more hyperparameter values from the original hyperparameter space to the
        normalized search space [0, 1]^k.

        Args:
            values (Union[object, List[object]]):
                Single value or list of values to normalize.

        Returns:
            normalized (ArrayLike):
                2D array of shape(len(values), self.K)
        """
        if not isinstance(values, (list, np.ndarray)):
            values = [values]

        values = [[value] if not isinstance(value, (list, np.ndarray)) else value
                  for value in values]

        values = np.array(values)
        self._within_hyperparam_space(values)

        return self._transform(values)
