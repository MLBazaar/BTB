# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

"""Package where the Tunable class is defined."""


class Tunable:
    """Tunable class.

    The Tunable class represents a collection of ``hyperparameters`` that need to be tuned as a
    whole, at once.

    Attributes:
        hyperparams:
            Dict of hyperparameters.
        names:
            List of names that the hyperparameters have and act as an ordering during the usage
            of ``inverse_transform`` method.
        SC:
            Int or ``np.inf`` amount that indicates the number of combinations possible for this
            tunable.

    Args:
        hyperparams (dict):
            Dictionary object that contains the name and the hyperparameter asociated to it.
    """

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.names = list(hyperparams)
        self.K = 0
        self.SC = 1

        for hyperparam in hyperparams.values():
            self.K = self.K + hyperparam.K
            self.SC = self.SC * hyperparam.SC

    def transform(self, values):
        """Transform one or more value combinations.

        Transform one or more value combinations from the original hyperparameters spaces to the
        normalized search space.

        Args:
            values (pandas.DataFrame, pandas.Series, dict, list(dict), 2D array-like):
                Values of shape ``(n, len(self.hyperparameters))``.

        Returns:
            numpy.ndarray:
                2D array of shape ``(len(values), K)`` where ``K`` is the sum of dimensions from
                all the ``hyperparameters`` that compose this ``tunable``.

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
        """Invert one or more value combinations.

        Invert one or more hyperparameter value combinations from the normalized search
        space :math:`[0, 1]^K` to the original hyperparameter space.

        Args:
            values (array-like):
                2D array of normalized values with shape ``(n, K)`` where ``K`` is the sum of
                dimensions from all the ``hyperparameters`` that compose this ``tunable``.

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
                item = value[:hyperparam.K]
                transformed.append(hyperparam.inverse_transform(item))
                value = value[hyperparam.K:]

            transformed = np.array(transformed, dtype=object)
            inverse_transform.append(np.concatenate(transformed, axis=1))

        return pd.DataFrame(np.concatenate(inverse_transform), columns=self.names)

    def sample(self, n_samples):
        """Sample values in the hyperparameters spaces for this tunable.

        Args:
            n_samlpes (int):
                Number of values to sample.

        Returns:
            numpy.ndarray:
                2D array with shape of ``(n_samples, K)`` where ``K``  is the sum of
                dimensions from all the ``hyperparameters`` that compose this ``tunable``.

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
