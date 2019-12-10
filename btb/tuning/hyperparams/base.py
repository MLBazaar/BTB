# -*- coding: utf-8 -*-

"""Package where the BaseHyperParam class is defined."""

from abc import ABCMeta, abstractmethod

import numpy as np


class BaseHyperParam(metaclass=ABCMeta):
    """BaseHyperParam class.

    A BaseHyperParam is an abstract representation of a single parameter that can be tuned.

    Attributes:
        dimensions (int):
            Number of dimensions that the hyperparameter uses to be represented in the search
            space.
        cardinality (int or np.inf):
            Number of possible values for this hyperparameter.
    """
    dimensions = 0
    cardinality = 0
    default = None

    def _to_array(self, values):
        """Validate and convert ``values`` to a ``numpy.ndarray`` with the expected shape.

        If ``self.dimensions`` is 1, only scalars, or lists of scalars or 1d arrays or
        2d arrays with one column are accepted and are transformed to:

            - scalar: 2d array with a single row and column.

            - list of scalars: 2d array with the list of scalars as its single column.

            - 1d array: 2d array with the given 1d array as its only column.

            - 2d array: the same array is returned unmodified.

        If ``self.dimensions`` is greater than one, only lists of scalars, or lists of lists
        or 1d arrays or 2d arrays are accepted and are transformed to:

            - list of scalars: 2d array with the list of scalars as its single row.

            - 1d array: 2d array with the given 1d array as its only row.

            - 2d array: the same array is returned unmodified.

        Args:
            values (scalar or array-like):
                A single scalar value or array-like of values to be converted to ``numpy.array``.

        Returns:
            numpy.ndarray:
                Values converted into ``numpy.ndarray`` with shape ``(, self.dimensions)``.

        Raises:
            ValueError:
                If the given values cannot fit into the expected output shape.
        """

        if np.isscalar(values):
            if self.dimensions == 1:
                return np.array([[values]])
            else:
                raise ValueError('Only lists or numpy.ndarrays are supported for dimensions > 1')

        if isinstance(values, list):
            if not all(np.isscalar(value) for value in values):
                if not all(isinstance(value, list) for value in values):
                    raise ValueError('Only list of lists are supported')
                elif not all(len(value) == self.dimensions for value in values):
                    raise ValueError('All sublists must have len == dimensions')

            values = np.array(values)

        if len(values.shape) > 2:
            raise ValueError('Invalid shape: Too many dimensions')

        if self.dimensions == 1:
            if len(values.shape) == 1:
                values = values.reshape(-1, 1)
            elif values.shape[1] != 1:
                raise ValueError('Invalid shape: Only 1 column is supported if dimensions == 1')

        elif len(values.shape) == 1:
            if len(values) != self.dimensions:
                raise ValueError('Number of elements != number of dimensions')
            elif not all(np.isscalar(value) for value in values):
                raise ValueError('Numpy arrays must only contain scalars')

            values = values.reshape(1, -1)

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
        self._within_range(values.astype(np.float), min=0, max=1)

    @abstractmethod
    def _inverse_transform(self, values):
        """Method to be implemented by child classes."""
        pass

    @abstractmethod
    def _transform(self, values):
        """Method to be implemented by child classes."""
        pass

    def inverse_transform(self, values):
        """Invert one or more search space values.

        Validates that the input values are within the search space and then transform them into
        hyperparameter values.

        Args:
            values (scalar or array-like):
                Scalar or array-like of values to be converted into the hyperparameter space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` containing values from the original hyperparameter space.

        Example:
            The example below shows simple usage case where an ``IntHyperParam`` is being imported,
            instantiated with a range from 1 to 4, and its method ``inverse_transform`` is being
            called two times with a scalar from the search space and an array of two valid values
            from the search space.

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
                2D ``numpy.ndarray`` with a shape `(n_samples, self.dimensions)`.

        Example:
            The example below shows simple usage case where an ``IntHyperParam`` is being imported,
            instantiated with a range from 1 to 4, and its method ``sample`` is being called
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

            - Scalar:
                A single scalar value from the original hyperparameter space.
            - List:
                A list composed by values from the original hyperparameter space.
            - 2D array-like:
                Two dimensions array-like object that contains values from the original
                hyperparameter space.

        Args:
            values (scalar, list or array-like):
                Single scalar value, list of values or array-like of values from the
                hyperparameter space to be converted into the search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` of shape `(len(values), self.dimensions)` containing the
                search space values.

        Example:
            The example below shows simple usage case where an ``IntHyperParam`` is being imported,
            instantiated with a range from 1 to 4, and its method ``transform`` is being called
            three times with a single scalar value, an array of two valid values and a 2D array
            with 1 dimension.

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
