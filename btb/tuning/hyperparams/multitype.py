# -*- coding: utf-8 -*-

"""Package where the MultitypeHyperParam class is defined."""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from btb.tuning.hyperparams import instantiate_hyperparam_from_dict
from btb.tuning.hyperparams.base import BaseHyperParam
from btb.tuning.hyperparams.boolean import BooleanHyperParam
from btb.tuning.hyperparams.categorical import CategoricalHyperParam
from btb.tuning.hyperparams.numerical import FloatHyperParam, IntHyperParam


class MultiTypeHyperparam(BaseHyperParam):
    r"""MultiTypeHyperParam Class.

    The MultitypeHyperParam class is responsible for the transformation of multiple value types
    in to normalized search space and provides the inverse transform from search space to
    hyperparameter space. Also provides a method that generates samples of those.

    Hyperparameter space:
        :math:`h_1, h_2,... h_K` where `K` is ``self.cardinality``.

    Search Space:
        :math:`\{ 0, 1 \}^K` where `K` is ``self.dimensions + len(self.configuration)``.

    Args:
        configuration (dict, list or tuple):
            - A python dictionary containing as ``key`` the given name for the hyperparameter
            and as value either an instance of a hyperparameter or a dictionary containg the
            following keys:
                - Type (str):
                    ``bool`` for ``BoolHyperParam``, ``int`` for ``IntHyperParam``, ``float``
                    for ``FloatHyperParam``, ``str`` for ``CategoricalHyperParam``.

                - Range or Values (list):
                    Range / values that this hyperparameter can take, in case of
                    ``CategoricalHyperParam`` those will be used as the ``choices``, for
                    ``NumericalHyperParams`` the ``min`` value will be used as the minimum
                    value and the ``max`` value will be used as the ``maximum`` value.

                - Default (str, bool, int, float or None):
                    The default value for the hyperparameter.

            - A python ``list`` or ``tuple`` containing the instances or a dictionaries as
            described previously.

        default (str or None):
            Default value for the hyperparameter to take. Defaults to the default value
            from the first ``element`` in ``self.configuration``.
    """

    NO_DEFAULT = object()  # dummy default in order to be able to set `None` as default

    def __init__(self, configuration, default=NO_DEFAULT):

        self.dimensions = 0
        self.cardinality = 0

        self.configuration = configuration
        if not isinstance(configuration, dict):
            self.configuration = dict(enumerate(configuration))

        for name, value in self.configuration.items():
            hyperparam = value

            if isinstance(value, dict):
                hyperparam = instantiate_hyperparam_from_dict(value)
                self.configuration[name] = hyperparam

            elif not isinstance(value, BaseHyperParam):
                raise TypeError('{} is not a hyperparameter based on'
                                '``btb.tuning.hyperparams.BaseHyperParam``'.format(value))

            self.dimensions += hyperparam.dimension
            self.cardinality += hyperparam.cardinality

        if default is self.NO_DEFAULT:
            self.default = self.configuration.values()[0].default

        self.choices = list(self.configuration.keys())
        _choices = np.array(self.choices, dtype='object')

        self._encoder = OneHotEncoder(categories=[_choices], sparse=False)
        self._encoder.fit(_choices.reshape(-1, 1))

    def _get_name_and_hyperparam(self, value):
        if isinstance(value, (list, np.ndarray)):
            value = value[0]

        if isinstance(value, (float, np.floating)):
            target_hyperparam_class = FloatHyperParam
        elif isinstance(value, (int, np.integer)):
            target_hyperparam_class = IntHyperParam
        elif isinstance(value, (bool, np.bool_)):
            target_hyperparam_class = BooleanHyperParam
        else:
            target_hyperparam_class = CategoricalHyperParam

        for name, hyperparam in self.configuration.items():
            if isinstance(hyperparam, target_hyperparam_class):
                return name, hyperparam

        return None, None

    def _within_hyperparam_space(self, values):
        """Validates the hyperparameter space of the values against their hyperparameters."""
        targets = [self._get_name_and_hyperparam(value)[1] for value in values]

        for target, value in zip(targets, values):
            target._within_hyperparam_space(value)

    def _unify(self, value, name):
        """Unify the output with the given dimensions.

        First encode the ``name`` of the hyperparameter of which this ``value`` pertains,
        then concatenate zeros to match the ``self.dimensions`` length. Finally concatenate
        the ``name`` encoded previously.

        Args:
            value (numpy.ndarray):
                A numpy.ndarray with normalized values inside the searc space :math:`{0, 1}`.
            name (str, int):
                A dictionary key from ``self.configuration``.

        Returns:
            numpy.ndarray:
                1D array with shape ``(1, self.dimensions + C)`` where ``C`` is the length of
                ``self.configuration``
        """
        name = np.array([[name]])
        name = self._encoder.transform(name.astype('object')).astype(int)
        zeros = self.dimensions - len(value[0])
        value = np.concatenate((value, np.zeros(zeros).reshape(1, -1)), axis=1)

        return np.concatenate((value, name), axis=1)[0]

    def _single_value_transform(self, value):
        """Transform a single value within the multiple hyperparameter spaces.

        Detect which hyperparameter instance reffers to this value and then
        transform and unify it in order to have the same dimension.
        """
        name, hyperparam = self._get_name_and_hyperparam(value)
        transformed = hyperparam.transform(value)
        return self._unify(transformed, name)

    def _transform(self, values):
        """Transform one or more multitype values.

        Transform one or more values that pertain to the multitype hyperparameter space in to
        the normalized search space of :math:`[0, 1]^K`.
        """
        transformed = np.array([
            self._single_value_transform(value)
            for value in values
        ])

        return transformed

    def _single_value_inverse_transform(self, value):
        """Invert a single value.

        Converts a 1D ``numpy.ndarray`` with a single value from the search space to the original
        hyperparamter space.

        Args:
            value (numpy.ndarray):
                1D array with values from the search space.

        Returns:
            numpy.ndarray:
                1D ``numpy.ndarray`` containing values from the original hyperparameter space.
        """
        name = np.array([value[-len(self.choices):]])
        name = self._encoder.inverse_transform(name.astype('object'))[0][0]
        hyperparam = self.configuration[name]

        return hyperparam.inverse_transform(value[:hyperparam.dimensions])[0]

    def _inverse_transform(self, values):
        """Invert one or more values.

        Converts a ``numpy.ndarray`` with normalized values from the search space to the original
        hyperparameter space.

        Args:
            values (numpy.ndarray):
                2D array with values from the search space.

        Returns:
            numpy.ndarray:
                2D ``numpy.ndarray`` containing values from the original hyperparameter space.
        """
        if len(values.shape) == 1:
            values = values.reshape(1, -1)

        inverted = np.array([
            self._single_value_inverse_transform(value)
            for value in values
        ])

        return inverted

    def sample(self, n_samples=1):
        """Generate sample values in the hyperparameter search space of ``[0, 1)``.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            numpy.ndarray:
                2D array with shape of ``(n_samples, 1)`` with normalized values inside the
                search space :math:`{0, 1}`.
        """
        indexes = np.random.random((n_samples, len(self.choices)))
        indexes = np.argmax(indexes, axis=1)
        choices = [self.choices[index] for index in indexes]

        samples = np.array([
            self._unify(self.configuration[choice].sample(1), choice)
            for choice in choices
        ])

        return samples
