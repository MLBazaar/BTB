# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from btb.tuning.hyperparams.boolean import BooleanHyperParam
from btb.tuning.hyperparams.categorical import CategoricalHyperParam
from btb.tuning.hyperparams.numerical import FloatHyperParam, IntHyperParam

"""Package where the Tunable class is defined."""


class Tunable:
    """Tunable class.

    The Tunable class represents a collection of ``HyperParams`` that need to be tuned as a
    whole, at once.

    Attributes:
        hyperparams:
            Dict of HyperParams.
        cardinality:
            Int or ``np.inf`` amount that indicates the number of combinations possible for this
            tunable.

    Args:
        hyperparams (dict):
            Dictionary object that contains the name and the hyperparameter asociated to it.
    """
    hyperparams = None
    names = None
    dimensions = 0
    cardinality = 1

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.names = list(hyperparams)

        for hyperparam in hyperparams.values():
            self.dimensions = self.dimensions + hyperparam.dimensions
            self.cardinality = self.cardinality * hyperparam.cardinality

    def transform(self, values):
        """Transform one or more hyperparameter configurations.

        Transform one or more hyperparameter configurations from the original hyperparameter
        space to the normalized search space.

        Args:
            values (pandas.DataFrame, pandas.Series, dict, list(dict), 2D array-like):
                Values of shape ``(n, len(self.hyperparams))``.

        Returns:
            numpy.ndarray:
                2D array of shape ``(len(values), dimensions)`` where ``dimensions`` is the sum of
                dimensions from all the ``HyperParams`` that compose this ``tunable``.

        Example:
            The example below shows a simple usage of a Tunable class which will transform a valid
            data from a 2D list and a ``numpy.ndarray`` is being returned.

            >>> from btb.tuning.hyperparams.boolean import BooleanHyperParam
            >>> from btb.tuning.hyperparams.categorical import CategoricalHyperParam
            >>> from btb.tuning.hyperparams.numerical import IntHyperParam
            >>> chp = CategoricalHyperParam(['cat', 'dog'])
            >>> bhp = BooleanHyperParam()
            >>> ihp = IntHyperParam(1, 10)
            >>> hyperparams = {
            ...     'chp': chp,
            ...     'bhp': bhp,
            ...     'ihp': ihp
            ... }
            >>> tunable = Tunable(hyperparams)
            >>> values = [
            ...     ['cat', False, 10],
            ...     ['dog', True, 1],
            ... ]
            >>> tunable.transform(values)
            array([[1.  , 0.  , 0.  , 0.95],
                   [0.  , 1.  , 1.  , 0.05]])
        """
        if isinstance(values, dict):
            values = pd.DataFrame([values])

        elif isinstance(values, list) and isinstance(values[0], dict):
            values = pd.DataFrame(values, columns=self.names)

        elif isinstance(values, list) and not isinstance(values[0], list):
            values = pd.DataFrame([values], columns=self.names)

        elif isinstance(values, pd.Series):
            values = values.to_frame().T

        elif not isinstance(values, pd.DataFrame):
            values = pd.DataFrame(values, columns=self.names)

        transformed = list()

        for name in self.names:
            hyperparam = self.hyperparams[name]
            value = values[name].values
            transformed.append(hyperparam.transform(value))

        return np.concatenate(transformed, axis=1)

    def inverse_transform(self, values):
        """Invert one or more hyperparameter configurations.

        Invert one or more hyperparameter configurations from the normalized search
        space :math:`[0, 1]^K` to the original hyperparameter space.

        Args:
            values (array-like):
                2D array of normalized values with shape ``(n, dimensions)`` where ``dimensions``
                is the sum of dimensions from all the ``HyperParams`` that compose this
                ``tunable``.

        Returns:
            pandas.DataFrame

        Example:
            The example below shows a simple usage of a Tunable class which will inverse transform
            a valid data from a 2D list and a ``pandas.DataFrame`` will be returned.

            >>> from btb.tuning.hyperparams.boolean import BooleanHyperParam
            >>> from btb.tuning.hyperparams.categorical import CategoricalHyperParam
            >>> from btb.tuning.hyperparams.numerical import IntHyperParam
            >>> chp = CategoricalHyperParam(['cat', 'dog'])
            >>> bhp = BooleanHyperParam()
            >>> ihp = IntHyperParam(1, 10)
            >>> hyperparams = {
            ...     'chp': chp,
            ...     'bhp': bhp,
            ...     'ihp': ihp
            ... }
            >>> tunable = Tunable(hyperparams)
            >>> values = [
            ...     [1, 0, 0, 0.95],
            ...     [0, 1, 1, 0.05]
            ... ]
            >>> tunable.inverse_transform(values)
               chp    bhp ihp
            0  cat  False  10
            1  dog   True   1
        """

        inverse_transform = list()

        for value in values:
            transformed = list()

            for name in self.names:
                hyperparam = self.hyperparams[name]
                item = value[:hyperparam.dimensions]
                transformed.append(hyperparam.inverse_transform(item))
                value = value[hyperparam.dimensions:]

            transformed = np.array(transformed, dtype=object)
            inverse_transform.append(np.concatenate(transformed, axis=1))

        return pd.DataFrame(np.concatenate(inverse_transform), columns=self.names)

    def sample(self, n_samples):
        """Sample values in the hyperparameters space for this tunable.

        Args:
            n_samlpes (int):
                Number of values to sample.

        Returns:
            numpy.ndarray:
                2D array with shape of ``(n_samples, dimensions)`` where ``dimensions``  is the
                sum of dimensions from all the ``HyperParams`` that compose this ``tunable``.

        Example:
            The example below shows a simple usage of a Tunable class which will generate 2
            samples by calling it's sample method. This will return a ``numpy.ndarray``.

            >>> from btb.tuning.hyperparams.boolean import BooleanHyperParam
            >>> from btb.tuning.hyperparams.categorical import CategoricalHyperParam
            >>> from btb.tuning.hyperparams.numerical import IntHyperParam
            >>> chp = CategoricalHyperParam(['cat', 'dog'])
            >>> bhp = BooleanHyperParam()
            >>> ihp = IntHyperParam(1, 10)
            >>> hyperparams = {
            ...     'chp': chp,
            ...     'bhp': bhp,
            ...     'ihp': ihp
            ... }
            >>> tunable = Tunable(hyperparams)
            >>> tunable.sample(2)
            array([[0.  , 1.  , 0.  , 0.45],
                   [1.  , 0.  , 1.  , 0.95]])
        """
        samples = list()

        for name, hyperparam in self.hyperparams.items():
            items = hyperparam.sample(n_samples)
            samples.append(items)

        return np.concatenate(samples, axis=1)

    def get_defaults(self):
        """Return the default combination for the hyperparameters."""
        return {
            name: hyperparam.default
            for name, hyperparam in self.hyperparams.items()
        }

    @classmethod
    def from_dict(cls, dict_hyperparams):
        """Create an instance from a dictionary containing information over hyperparameters.

        Class method that creates an instance from a dictionary that describes the type of a
        hyperparameter, the range or values that this can have and the default value of the
        hyperparameter.

        Args:
            dict_hyperparams (dict):
                A python dictionary containing as `key` the given name for the hyperparameter and
                as value a dictionary containing the following keys:

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

        Returns:
            Tunable:
                A ``Tunable`` instance with the given hyperparameters.
        """

        if not isinstance(dict_hyperparams, dict):
            raise TypeError('Hyperparams must be a dictionary.')

        hyperparams = {}

        for name, hyperparam in dict_hyperparams.items():
            hp_type = hyperparam['type']
            hp_default = hyperparam.get('default')

            if hp_type == 'int':
                hp_range = hyperparam.get('range') or hyperparam.get('values')
                hp_min = min(hp_range) if hp_range else None
                hp_max = max(hp_range) if hp_range else None
                hp_instance = IntHyperParam(min=hp_min, max=hp_max, default=hp_default)

            elif hp_type == 'float':
                hp_range = hyperparam.get('range') or hyperparam.get('values')
                hp_min = min(hp_range)
                hp_max = max(hp_range)
                hp_instance = FloatHyperParam(min=hp_min, max=hp_max, default=hp_default)

            elif hp_type == 'bool':
                hp_instance = BooleanHyperParam(default=hp_default)

            elif hp_type == 'str':
                hp_choices = hyperparam.get('range') or hyperparam.get('values')
                hp_instance = CategoricalHyperParam(choices=hp_choices, default=hp_default)

            hyperparams[name] = hp_instance

        return cls(hyperparams)

    def __repr__(self):
        return 'Tunable({})'.format(self.hyperparams)
