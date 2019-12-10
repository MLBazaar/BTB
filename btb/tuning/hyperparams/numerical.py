# -*- coding: utf-8 -*-

"""Package where the NumericalHyperParam class and it's childs are defined."""

import sys

import numpy as np

from btb.tuning.hyperparams.base import BaseHyperParam


class NumericalHyperParam(BaseHyperParam):
    """NumericalHyperParam class.

    The NumericalHyperParam class defines an abstraction to hyperparameters which ranges are
    defined by a numerical value and can take any number within that range.
    """

    dimensions = 1

    def _within_range(self, values, min=0, max=1):
        if (values < min).any() or (values > max).any():
            raise ValueError('Value not within range [{}, {}]: {}'.format(min, max, values))


class FloatHyperParam(NumericalHyperParam):
    """FloatHyperParam class.

    The FloatHyperParam class represents a single hyperparameter within a range of ``float``
    numbers, where ``min`` and ``max`` can take as value any float number within that range,
    having ``min`` to be smaller than ``max``.

    Hyperparameter space:
        :math:`h_1, h_2,... h_n` where :math:`h_i = i * (max - min) + min`

    Search space:
        :math:`s_1, s_2,... s_n` where :math:`s_i = (i - min) / (max - min)`

    Args:
        min (float):
            Float number to represent the minimum value that this hyperparameter can take,
            by default is ``None`` which will take the system's minimum float value possible.

        max (float):
            Float number to represent the maximum value that this hyperparameter can take,
            by default is ``None`` which will take the system's maximum float value possible.

        default (float):
            Float number that represents the default value for the hyperparameter. Defaults to
            ``self.min``

        include_min (bool):
            Either or not to include the minimum value in the search space.

        include_max (bool):
            Either or not to include the maximum value in the search space.
    """

    cardinality = np.inf

    def __init__(self, min=None, max=None, default=None, include_min=True, include_max=True):

        self.include_min = include_min
        self.include_max = include_max

        if min is None or min == -np.inf:
            min = sys.float_info.min

        if max is None or max == np.inf:
            max = sys.float_info.max

        if min >= max:
            raise ValueError('The ``min`` value can not be greater or equal to ``max`` value.')

        if default is None:
            self.default = float(min)
        else:
            self.default = float(default)

        self.min = float(min)
        self.max = float(max)
        self.range = max - min

    def _inverse_transform(self, values):
        """Invert one or more search space values.

        Converts a ``numpy.ndarray`` with normalized values from the search space :math:`s_1,
        s_2,... s_n` to the hyperparameter space of :math:`h_1, h_2,... h_n`. This is
        being computed by multiplying the hyperparameter's range with the values to be denormalized
        and adding the ``self.min`` value to them.

        Args:
            values (numpy.ndarray):
                2D array with values from the search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` containing values from the original hyperparameter space.

        Example:
            The example below shows simple usage case where a FloatHyperParam is being created
            with a range that goes from ``0.1`` to ``0.9`` and it's ``_inverse_transform`` method
            is being called with a valid ``numpy.ndarray`` that contain values from the normalized
            search space and values from the hyperparameter space are being returned.

            >>> instance = FloatHyperParam(min=0.1, max=0.9)
            >>> instance._inverse_transform(np.array([[0.], [1.]]))
            array([[0.1],
                   [0.9]])
            >>> instance._inverse_transform(np.array([[0.875]]))
            array([[0.8]])
        """
        return values * self.range + self.min

    def _transform(self, values):
        """Transform one or more hyperparameter values.

        Converts a ``numpy.ndarray`` with ``float`` values from the original hyperparameter space
        :math:`h_1, h_2,... h_n` into the normalized search space :math:`s_1, s_2,... s_n`.
        This is being computed by substracting the ``self.min`` value from the values to be
        trasnformed and dividing them by the ``self.range`` of the hyperparameter.

        Args:
            values (numpy.ndarray):
                2D array with values from the hyperparameter space to be converted into the
                search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` of shape `(len(values), 1)` containing the search space
                values.

        Example:
            The example below shows simple usage case where a FloatHyperParam is being created
            with a range that goes from ``0.1`` to ``0.9`` and it's ``_transform`` method
            is being called with a valid ``numpy.ndarray`` that contain values from the
            hyperparameter space and values from the search space are being returned.

            >>> instance = FloatHyperParam(min=0.1, max=0.9)
            >>> instance._transform(np.array([[0.1], [0.9]]))
            array([[0.],
                   [1.]])
            >>> instance._transform(np.array([[0.8]]))
            array([[0.875]])
        """
        return (values - self.min) / self.range

    def sample(self, n_samples):
        """Generate sample values in the hyperparameter search space :math:`{0, 1}`.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            numpy.ndarray:
                2D array with shape of `(n_samples, 1)` with normalized values inside the
                search space :math:`{0, 1}`.

        Example:
            The example below shows simple usage case where a FloatHyperParam is being created
            with a range that goes from ``0.1`` to ``0.9`` and it's ``sample`` method
            is being called with a number of samples to be obtained. A ``numpy.ndarray`` with
            values from the search space is being returned.

            >>> instance = FloatHyperParam(min=0.1, max=0.9)
            >>> instance.sample(2)
            array([[0.52058728],
                   [0.00582452]])
        """
        return np.random.random((n_samples, self.dimensions))

    def __repr__(self):
        args = (self.min, self.max, self.default, self.include_min, self.include_max)
        args = 'min={}, max={}, default={}, include_min={}, include_max={}'.format(*args)
        return 'FloatHyperParam({})'.format(args)


class IntHyperParam(NumericalHyperParam):
    """IntHyperParam class.

    The IntHyperParam class represents a single hyperparameter within an range of ``int``
    numbers, where ``min`` and ``max`` can take as value any ``int`` number that compose this range
    having ``min`` to be smaller than ``max``.

    Hyperparameter space:
        :math:`h_1, h_2,... h_n` where :math:`h_i = min + (i - 1) * step`

    Search space:
        :math:`s_1, s_2,... s_n` where :math:`s_i = \\frac{interval}{2} + (i - 1) * interval`

    Args:
        min (int):
            Integer number to represent the minimum value that this hyperparameter can take,
            by default is ``None`` which will take the system's minimum int value possible.

        max (int):
            Integer number to represent the maximum value that this hyperparameter can take,
            by default is ``None`` which will take the system's maximum int value possible.

        default (int):
            Integer number that represents the default value for the hyperparameter. Defaults to
            ``self.min``.

        step (int):
            Increase amount to take for each sample. Defaults to 1.

        include_min (bool):
            Either or not to include the minimum value in the search space.

        include_max (bool):
            Either or not to include the maximum value in the search space.
    """

    dimensions = 1

    def __init__(self, min=None, max=None, default=None,
                 include_min=True, include_max=True, step=1):

        self.include_min = include_min
        self.include_max = include_max

        if min is None or min == -np.inf:
            min = -(sys.maxsize / 2)

        if max is None or max == np.inf:
            max = sys.maxsize / 2

        if min >= max:
            raise ValueError('The `min` value can not be greater or equal to `max` value.')

        if default is None:
            self.default = min
        else:
            self.default = int(default)

        self.min = int(min) if include_min else int(min) + 1
        self.max = int(max) if include_max else int(max) - 1
        self.step = step
        self.cardinality = ((self.max - self.min) // step) + 1

        if (self.max - self.min) % self.step:
            raise ValueError(
                "Invalid step of {} for values inside [{}, {}]".format(step, self.min, self.max)
            )

        self.interval = self.step / (self.max - self.min + self.step)

    def _inverse_transform(self, values):
        """Invert one or more search space values.

        Converts a ``numpy.ndarray`` with normalized values from the search space :math:`s_1,
        s_2,... s_n` to the hyperparameter space of :math:`h_1, h_2,... h_n`. This is
        being computed by divinding the hyperparameter's interval with the values to be inverted
        and adding the ``self.min`` value to them and resting the 0.5 that has been added during
        the transformation.

        Args:
            values (numpy.ndarray):
                2D array with values from the search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` containing values from the original hyperparameter space.

        Example:
            The example below shows simple usage case where an IntHyperParam is being created
            with a range that goes from ``1`` to ``4`` and it's ``_inverse_transform`` method
            is being called with a valid ``numpy.ndarray`` that contain values from the
            search space and values from the hyperparameter space are being returned.

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
        """Transform one or more hyperparameter values.

        Converts a ``numpy.ndarray`` with ``int`` values from the original hyperparameter space
        :math:`h_1, h_2,... h_n` into the normalized search space :math:`s_1, s_2,... s_n`.
        This is being computed by substracting the ``min`` value that the hyperparameter can take
        from the values to be trasnformed and adding them 0.5, then multiply by the interval.

        Args:
            values (numpy.ndarray):
                2D array with values from the hyperparameter space to be converted into the
                search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` of shape `(len(values), 1)` containing the search space
                values.

        Example:
            The example below shows simple usage case where an IntHyperParam is being created
            with a range that goes from ``1`` to ``4`` and it's ``_transform`` method
            is being called with a valid ``numpy.ndarray`` that contain values from the
            hyperparameter space and values from the search space are being returned.

            >>> instance = IntHyperParam(min=1, max=4)
            >>> instance._transform(np.array([[1], [4]]))
            array([[0.125],
                   [0.875]])
            >>> instance._trasnfrom(np.array([[3]]))
            array([[0.625]])
        """
        return ((values - self.min) / self.step + 0.5) * self.interval

    def sample(self, n_samples):
        """Generate sample values in the hyperparameter search space of [0, 1).

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            numpy.ndarray:
                2D array with shape of `(n_samples, 1)` with normalized values inside the
                search space :math:`{0, 1}`.

        Example:
            The example below shows simple usage case where a IntHyperParam is being created
            with a range that goes from ``1`` to ``4`` and it's ``sample`` method
            is being called with a number of samples to be obtained. A ``numpy.ndarray`` with
            values from the search space is being returned.

            >>> instance = IntHyperParam(min=1, max=4)
            >>> instance.sample(2)
            array([[0.625],
                   [0.375]])
        """
        sampled = np.random.random((n_samples, self.dimensions))
        inverted = self._inverse_transform(sampled)

        return self._transform(inverted)

    def __repr__(self):
        args = (self.min, self.max, self.default, self.include_min, self.include_max, self.step)
        args = 'min={}, max={}, default={}, include_min={}, include_max={}, step={}'.format(*args)
        return 'IntHyperParam({})'.format(args)
