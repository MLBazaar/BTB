# -*- coding: utf-8 -*-

"""Package where the BaseHyperParam class is defined."""

from abc import ABCMeta, abstractmethod

import numpy as np


class BaseHyperParam(metaclass=ABCMeta):
    """BaseHyperParam class.

    BaseHyperParam class is abstract representation of a single hyperparameter that can
    be tuned.

    Attributes:
        K (int):
            Number of dimensions that the hyperparameter uses to be represented in the search
            space.
    """

    def _to_array(self, values):
        """Validate values and convert them to ``numpy.array`` with ``self.K`` dimension/s.

        Perform a validation over ``values`` to ensure that it can be converted to a valid
        hyperparameter space or search space, then convert the given values to a ``numpy.array``
        of ``self.K`` dimension/s.

        Args:
            values (single value or array-like):
                A sinlge value or array-like of values to be converted to ``numpy.array``.

        Returns:
            numpy.ndarray:
                Values converted into ``numpy.ndarray`` with shape ``(n, dimensions)`` where
                ``n`` is the length of values and ``dimensions`` is ``self.K``.

        Raises:
            ValueError:
                A ``ValueError`` is raised if any value from ``values`` is not represented in the
                ``self.K`` dimension/s or the shape has more than two dimensions.
        """

        if self.K > 1:
            if not isinstance(values, (list, np.ndarray)):
                raise ValueError(
                    'Value: {} is not valid for {} dimensions.'.format(values, self.K)
                )

            elif not isinstance(values[0], (list, np.ndarray)):
                values = [values]

        else:
            if isinstance(values, (list, np.ndarray)):
                values = [
                    value if isinstance(value, (list, np.ndarray)) else [value]
                    for value in values
                ]

            else:
                values = [[values]]

        if not all(len(value) == self.K for value in values):
            raise ValueError(
                'All the values must be {} dimension/s.'.format(self.K)
            )

        values = np.array(values)

        if len(values.shape) > 2:
            raise ValueError('Only shapes of 1 or 2 dimensions are supported.')

        return values

    def _within_range(self, values, min=0, max=1):
        """Ensure that the values are within a certain range.

        Args:
            values (numpy.ndarray):
                2D array of values that will be validated.

        Raises:
            ValueError:
                A ``ValueError`` is raised if any value from ``values`` is not within the range.
        """
        inside_mask = np.ma.masked_inside(values, min, max)
        if inside_mask.any():
            outside = inside_mask[~inside_mask.mask].data.reshape(-1).tolist()
            raise ValueError(
                'Values found outside of the valid range [{}, {}]: {}'.format(min, max, outside)
            )

    def _within_hyperparam_space(self, values):
        """Ensure that the values are within the range of the hyperparameter space.

        Args:
            values (numpy.ndarray):
                2D array of values that will be validated.
        """
        self._within_range(values, min=self.min, max=self.max)

    def _within_search_space(self, values):
        """Ensure that the values are within the range of the search space.

        Args:
            values (numpy.ndarray):
                2D array of values that will be validated.
        """
        self._within_range(values, min=0, max=1)

    @abstractmethod
    def _inverse_transform(self, values):
        pass

    @abstractmethod
    def _transform(self, values):
        pass

    def inverse_transform(self, values):
        """Invert one or more search space values.

        Validates that the input values are within the search space and then transform them into
        hyperparameter values.

        Args:
            values (single value or array-like):
                Single value or array-like of values to be converted into the hyperparameter space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` containing values from the original hyperparameter space.

        Example:
            The example below shows simple usage case where an ``IntHyperParam`` is being imported,
            instantiated with a range from 1 to 4, and it's method ``inverse_transform`` is being
            called two times with a single value from the search space and an array of two valid
            values from the search space.

            >>> from btb.tuning.hyperparams.numerical import IntHyperParam
            >>> ihp = IntHyperParam(min=1, max=4)
            >>> ihp.inverse_transform(0.125)
            array([[1]])
            >>> ihp.inverse_transform([0.125, 0.375])
            array([[1],
                   [2]])
        """
        values = self._to_array(values)
        self._within_search_space(values)

        return self._inverse_transform(values)

    @abstractmethod
    def sample(self, n_samples):
        """Generate an array of ``n_samples`` random samples in the search space.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` with a shape `(n_samples, self.K)`.

        Example:
            The example below shows simple usage case where an ``IntHyperParam`` is being imported,
            instantiated with a range from 1 to 4, and it's method ``sample`` is being called
            with a number of samples to be obtained. A ``numpy.ndarray`` with values from the
            search space is being returned.

            >>> from btb.tuning.hyperparams.numerical import IntHyperParam
            >>> instance = IntHyperParam(min=1, max=4)
            >>> instance.sample(2)
            array([[0.625],
                   [0.375]])
        """

    def transform(self, values):
        """Transform one or more hyperparameter values.

        Validates that the input values are within the accepted dimensions and that they are within
        the hyperparameter space. Then transform one or more hyperparameter values from the
        original hyperparameter space into the normalized search space :math:`[0, 1]^K`.
        The accepted value formats are:

            - Single value:
                A single value from the original hyperparameter space.
            - List:
                A list composed by values from the original hyperparameter space.
            - 2D array-like:
                Two dimensions array-like object that contains values from the original
                hyperparameter space.

        Args:
            values (single value, list or array-like):
                Single value, list of values or array-like of values from the hyperparameter space
                to be converted into the search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` of shape `(len(values), self.K)` containing the search space
                values.

        Example:
            The example below shows simple usage case where an ``IntHyperParam`` is being imported,
            instantiated with a range from 1 to 4, and it's method ``transform`` is being called
            three times with a single value, array of two valid values and 2D array of 1 dimension.

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
