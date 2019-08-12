# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class Tunable:
    """Tunable class.

    Collection of HyperParams that need to be tuned as a whole.

    Attributes:
        hyperparams: List of hyperparameters from this hyperparam space.
    """

    def __init__(self, hyperparams, names=None):
        """Creates an instance of a Tunable class.

        Args:
            hyperparams(dictLike):
                Dictionary like object that contains the name and the hyperparameter asociated to
                it.
            names(list):
                List of names to be used as order during inverse_transform. If this value is None,
                the default order from the dictionary will be used. You should check ``self.names``
                so you can give the correct order of your transformed data.
        """

        self.hyperparams = hyperparams

        if names is None:
            names = list(hyperparams)

        self.names = names

    def transform(self, values):
        """Transform one or more hyperparameter value combinations.

        Transform one or more hyperparameter value combinations from the original hyperparameter
        space the normalized search space [0, 1]^K.

        Args:
            values (pandas.DataFrame, pandas.Series, dict, list(dict), 2D ArrayLike):
                Values of shape (*, len(self.hyperparameters)).

        Returns:
            transformed (ArrayLike):
                2D array of shape (len(values), K)
        """
        if isinstance(values, dict):
            values = pd.DataFrame([values])
        elif isinstance(values, list) and isinstance(values[0], dict):
            values = pd.DataFrame(values, columns=self.names)
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
        """Inverse transform one or more hyperparameter value combinations.

        Transform one or more hyperparameter values from the normalized search space [0, 1]^K to
        the original hyperparameter space.

        Args:
            values (ArrayLike):
                2D array of normalized values with shape (*, K).

        Returns:
            pandas.DataFrame
        """

        inverse_transform = list()

        for value in values:
            transformed = list()

            for name in self.names:
                hyperparam = self.hyperparams[name]
                item = value[:hyperparam.K]
                transformed.append(hyperparam.inverse_transform(item))
                value = value[hyperparam.K:]

            transformed = np.array(transformed, dtype=object)  # perserve the original dtypes
            inverse_transform.append(np.concatenate(transformed, axis=1))

        return pd.DataFrame(np.concatenate(inverse_transform), columns=self.names)

    def sample(self, n_samples):
        """Generate sample values for this hyperparameters.

        Args:
            n_samlpes (int):
                Number of values to sample.

        Returns:
            samples (ArrayLike):
                2D array with shape of (n_samples, sum(hyperparams.K)).
        """
        samples = list()

        for name, hyperparam in self.hyperparams.items():
            items = hyperparam.sample(n_samples)
            samples.append(items)

        return np.concatenate(samples, axis=1)

    def to_dict(self):
        """Get a dict representation of this Tunable."""
        pass

    @classmethod
    def from_dict(cls, spec_dict):
        """Load a Tunable from a dict representation."""
        pass
