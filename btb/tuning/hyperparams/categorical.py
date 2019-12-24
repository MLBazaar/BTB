# -*- coding: utf-8 -*-

"""Package where the CategoricalHyperParamClass is defined."""

from copy import deepcopy

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from btb.tuning.hyperparams.base import BaseHyperParam


class CategoricalHyperParam(BaseHyperParam):
    r"""CategoricalHyperParam Class.

    The CategoricalHyperParam class is responsible for the transform of categorical values
    in to normalized search space and provides the inverse transform from search space to
    hyperparameter space. Also provides a method that generates samples of those.

    Hyperparameter space:
        :math:`h_1, h_2,... h_K` where `K` is the number of categories.

    Search Space:
        :math:`\{ 0, 1 \}^K` where `K` is the number of categories.

    Args:
        choices (list):
            List of values that the hyperparameter can be.

        default (str or None):
            Default value for the hyperparameter to take. Defaults to the first item in ``choices``
    """
    NO_DEFAULT = object()

    def __init__(self, choices, default=NO_DEFAULT):
        """Instantiation of CategoricalHyperParam.

        Creates an instance with a list of ``choices`` and fit an instance of
        ``sklearn.preprocessing.OneHotEncoder`` with those values.
        """
        if default is self.NO_DEFAULT:
            self.default = choices[0]
        elif default not in choices:
            raise ValueError('`default` not within `choices`')
        else:
            self.default = default

        self.choices = deepcopy(choices)
        self.dimensions = len(choices)
        self.cardinality = self.dimensions
        choices = np.array(choices, dtype='object')
        self._encoder = OneHotEncoder(categories=[choices], sparse=False)
        self._encoder.fit(choices.reshape(-1, 1))

    def _within_hyperparam_space(self, values):
        mask = np.isin(values, self.choices)

        if not mask.all():
            if not isinstance(values, np.ndarray):
                values = np.asarray(values)

            not_in_space = values[~mask].tolist()
            raise ValueError(
                'Values found outside of the valid space {}: {}'.format(self.choices, not_in_space)
            )

    def _inverse_transform(self, values):
        """Invert one or more values.

        Converts a ``numpy.ndarray`` with normalized values from the search space to the original
        hyperparameter space of ``self.choices``.

        Args:
            values (numpy.ndarray):
                2D array with values from the search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` containing values from the original hyperparameter space.

        Example:
            The example below shows simple usage case where a CategoricalHyperParam is being
            created with three possible values, (Cat, Dog, Tiger), and it's method
            ``_inverse_transform`` is being called with a valid ``numpy.ndarray`` containing
            values from the search space and values from the hyperparameter space are being
            returned.

            >>> instance = CategoricalHyperParam(choices=['Cat', 'Dog', 'Tiger'])
            >>> instance._inverse_transform(np.array([[1, 0, 0]]))
            array([['Cat']])
            >>> instance._inverse_transform(np.array([[1, 0, 0], [0, 0, 1]]))
            array([['Cat'],
                   ['Tiger']])
        """
        if len(values.shape) == 1:
            values = values.reshape(1, -1)

        return self._encoder.inverse_transform(values.astype('object'))

    def _transform(self, values):
        """Transform one or more categorical values.

        Encodes one or more categorical values in to the normalized search space of
        :math:`[0, 1]^K` by using ``sklearn.preprocessing.OneHotEncoder`` that has been
        fitted during the instantiation.

        Args:
            values (numpy.ndarray):
                2D array with values from the hyperparameter space to be converted into the
                search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` of shape `(len(values), self.dimensions)` containing the
                search space values.

        Example:
            The example below shows simple usage case where a CategoricalHyperParam is being
            created with three possible values, (Cat, Dog, Tiger), and it's method ``_transform``
            is being called with a valid ``numpy.ndarray`` containing values from the
            hyperparameter space and an array with normalized values is being returned.

            >>> instance = CategoricalHyperParam(choices=['Cat', 'Dog', 'Tiger'])
            >>> instance._transform(np.array([['Cat']]))
            array([[1, 0, 0]])
            >>> instance._transform(np.array([['Cat'], ['Tiger']]))
            array([[1, 0, 0],
                   [0, 0, 1]])
        """
        return self._encoder.transform(values.astype('object')).astype(int)

    def sample(self, n_samples):
        """Generate sample values in the hyperparameter search space of ``[0, 1]^K``.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            numpy.ndarray:
                2D array with shape of `(n_samples, self.dimensions)` with normalized values
                inside the search space :math:`[0, 1]^K`.

        Example:
            The example below shows simple usage case where a CategoricalHyperParam is being
            created with three possible values, (Cat, Dog, Tiger), and it's method ``sample``
            is being called with a number of samples to be obtained. A ``numpy.ndarray`` with
            values from the search space is being returned.

            >>> instance = CategoricalHyperParam(choices=['Cat', 'Dog', 'Tiger'])
            >>> instance.sample(2)
            array([[1, 0, 0],
                   [0, 1, 0]])
        """
        randomized_values = np.random.random((n_samples, self.dimensions))
        indexes = np.argmax(randomized_values, axis=1)

        sampled = [self.choices[index] for index in indexes]

        return self.transform(sampled)

    def __repr__(self):
        return 'CategoricalHyperParam(choices={}, default={})'.format(self.choices, self.default)
