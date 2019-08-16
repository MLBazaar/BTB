# -*- coding: utf-8 -*-

"""Package where the numerical hyperparameters are defined."""

import sys

import numpy as np

from btb.tuning.hyperparams.base import BaseHyperParam


class NumericalHyperParam(BaseHyperParam):
    """Numerical Hyperparameter Class.

    The numerical hyperparameter class defines an abstraction to hyperparameters which ranges are
    defined by a numerical value and can take any number between that range.

    Attributes:
        K (int):
            Number of dimensions that this hyperparameter uses to be represented in the search
            space.

    Args:
        min (int or float):
            Minimum numerical value that this hyperparameter can be set to.
        max (int or float):
            Maximum numerical value that this hyperparameter can be set to.
        include_min (default=True):
            Either ot include or not the ``min`` value inside the range.
        include_max (default=True):
            Either ot include or not the ``max`` value inside the range.
        step (int or float):
            Increase amount to take for each sample.
    """


class FloatHyperParam(NumericalHyperParam):
    """NumericalHyperParam of type ``float``.

    Hyperparameter space:
        ``[min, max]``

    Search space:
        ``[0, 1)``
    """

    K = 1

    def __init__(self, min=None, max=None, include_min=True, include_max=True):

        self.include_min = include_min
        self.include_max = include_max

        if min is None or min == -np.inf:
            min = sys.float_info.min

        if max is None or max == np.inf:
            max = sys.float_info.max

        if min >= max:
            raise ValueError('The `min` value can not be greater or equal to `max` value.')

        self.min = min
        self.max = max
        self.range = max - min

    def _inverse_transform(self, values):
        """Invert one or more ``normalized`` values.

        Inverse transorm one or more normalized values from the search space [0, 1)^k. This is
        being computed by multiplying the hyperparameter's range with the values to be denormalized
        and adding the ``self.min`` value to them.

        Args:
            values (numpy.ndarray):
                2D array of normalized values.

        Returns:
            denormalized (numpy.ndarray):
                2D array of denormalized values.

        Examples:
            >>> instance = FloatHyperParam(min=0.1, max=0.9)
            >>> instance._inverse_transform(np.array([[0.], [1.]]))
            array([[0.1],
                   [0.9]])
            >>> instance._inverse_transform(np.array([[0.875]]))
            array([[0.8]])
        """
        return values * self.range + self.min

    def _transform(self, values):
        """Transform one or more ``float`` values.

        Convert one or more ``float`` values from the original hyperparameter space to the
        normalized search space [0, 1)^k. This is being computed by substracting the ``self.min``
        value that the hyperparameter can take from the values to be trasnformed and dividing them
        by the ``self.range`` of the hyperparameter.

        Args:
            values (numpy.ndarray):
                2D array with values to be normalized.

        Returns:
            normalized (numpy.ndarray):
                2D array of shape(len(values), self.K).

        Examples:
            >>> instance = FloatHyperParam(min=0.1, max=0.9)
            >>> instance._transform(np.array([[0.1], [0.9]])
            array([[0.],
                   [1.]])
            >>> instance._transform(np.array([[0.8]])
            array([[0.875]])
        """
        return (values - self.min) / self.range

    def sample(self, n_samples):
        """Generate sample values in the hyperparameter search space of [0, 1)^k.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            samples (numpy.ndarray):
                2D array with shape of (n_samples, self.K) with normalized values inside the search
                space [0, 1)^k.

        Examples:
            >>> instance = FloatHyperParam(min=0.1, max=0.9)
            >>> instance.sample(2)
            array([[0.52058728],
                   [0.00582452]])
        """
        return np.random.random((n_samples, self.K))


class IntHyperParam(NumericalHyperParam):
    """NumericalHyperParam of type ``int``.

    Hyperparameter space:
        ``{min, min + step, min + 2 * step...max}``
    Search space:
        ``{i1...in} where n is ((max - min) / step) + 1``
    """

    K = 1

    def __init__(self, min=None, max=None, include_min=True, include_max=True, step=1):

        self.include_min = include_min
        self.include_max = include_max

        if min is None:
            min = -(sys.maxsize / 2)

        if max is None:
            max = sys.maxsize / 2

        if min >= max:
            raise ValueError('The `min` value can not be greater or equal to `max` value.')

        self.min = int(min) if include_min else int(min) + 1
        self.max = int(max) if include_max else int(max) - 1
        self.step = step
        self.range = ((self.max - self.min) / step) + 1
        self.interval = 1 / self.range

    def _inverse_transform(self, values):
        """Invert one or more ``normalized`` values.

        Inverse transorm one or more normalized values from the search space [0, 1)^k. This is
        being computed by divinding the hyperparameter's interval with the values to be inverted
        and adding the ``self.min`` value to them and resting the 0.5 that has been added during
        the transformation.

        Args:
            values (numpy.ndarray):
                2D array of normalized values.

        Returns:
            denormalized (numpy.ndarray):
                2D array of denormalized values.

        Examples:
            >>> instance = IntHyperParam(min=1, max=4)
            >>> instance._inverse_transform(np.array([[0.125], [0.875]]))
            array([[1],
                   [4]])
            >>> instance._inverse_trasnfrom(np.array([[0.625]]))
            array([[3]])
        """
        unscaled_values = values / self.interval - 0.5 + self.min
        rounded = unscaled_values.round()

        # Restrict to make sure that we stay within the valid range
        restricted = np.minimum(np.maximum(rounded, self.min), self.max)

        return restricted.astype(int)

    def _transform(self, values):
        """Transform one or more ``int`` values.

        Convert one or more ``int`` values from the original hyperparameter space to the
        normalized search space [0, 1)^k. This is being computed by substracting the `min` value
        that the hyperparameter can take from the values to be trasnformed and adding them 0.5,
        then multiply by the interval.

        Args:
            values (numpy.ndarray):
                2D array of values to be normalized.

        Returns:
            normalized (numpy.ndarray):
                2D array of shape(len(values), self.K).

        Examples:
            >>> instance = IntHyperParam(min=1, max=4)
            >>> instance._transform(np.array([[1], [4]]))
            array([[0.125],
                   [0.875]])
            >>> instance._trasnfrom(np.array([[3]])
            array([[0.625]])
        """
        return (values - self.min + 0.5) * self.interval

    def sample(self, n_samples):
        sampled = np.random.random((n_samples, self.K))
        inverted = self._inverse_transform(sampled)

        return self._transform(inverted)
