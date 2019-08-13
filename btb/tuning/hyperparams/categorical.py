# -*- coding: utf-8 -*-

"""Package where the categorical hyperparameters are defined."""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from btb.tuning.hyperparams.base import BaseHyperParam


class CategoricalHyperParam(BaseHyperParam):
    """Categorical Hyperparameter Class.

    The categorical hyperparameter class it is responsible for the transform of categorical values
    in to normalized search space of [0, 1]^K and provides the inverse transform from search space
    [0, 1]^K to hyperparameter space. Also provides a method that generates samples of those.

    Hyperparameter space:
        ``{c1...cK}`` where K is the number of categories

    Search Space:
        ``{0, 1}^K``

    Attributes:
        K (int):
            Number of dimensions that this hyperparameter uses to be represented in the search
            space.

    Args:
        choices (list):
            choices (List[object]):
                List of values that the hyperparameter can be.
    """

    def __init__(self, choices):
        self.choices = choices
        self.K = len(choices)
        self._encoder = OneHotEncoder(sparse=False)
        self._encoder.fit(np.array(choices).reshape(-1, 1))

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
        """Invert one or more values from search space [0, 1]^K.

        Converts normalized ``values`` from the search space [0, 1]^K to the original space of
        ``choices`` that this hyperparameter has been instantiated with.

        Args:
            values (ArrayLike):
                Single value or 2D ArrayLike of normalized values.

        Returns:
            denormalized (Union[object, List[object]]):
                Denormalized value or list of denormalized values.

        Examples:
            >>> import numpy
            >>> instance = CategoricalHyperParam(choices=['Cat', 'Dog', 'Tiger'])
            >>> instance._inverse_transform(numpy.asarray([[1, 0, 0]]))
            array([['Cat']])
            >>> instance._inverse_transform(numpy.asarray([[1, 0, 0], [0, 0, 1]]))
            array([['Cat'],
                   ['Tiger']])
        """

        if len(values.shape) == 1:
            values = values.reshape(1, -1)

        return self._encoder.inverse_transform(values)

    def _transform(self, values):
        """Transform one or more categorical values.

        Encodes one or more categorical values in to the normalized search space of [0, 1]^k
        by using ``sklearn.preprocessing.OneHotEncoder`` that has been fitted during the
        initialization.

        Args:
            values (ArrayLike):
                2D ArrayLike of normalized values.

        Returns:
            normalized (ArrayLike):
                2D array of shape(len(values)).

        Examples:
            >>> import numpy
            >>> instance = CategoricalHyperParam(choices=['Cat', 'Dog', 'Tiger'])
            >>> instance._transform(numpy.asarray([['Cat']])
            array([[1, 0, 0]])
            >>> instance._transform(numpy.asarray([['Cat'], ['Tiger']])
            array([[1, 0, 0],
                   [0, 0, 1]])
        """
        return self._encoder.transform(values.reshape(-1, 1)).astype(int)

    def sample(self, n_samples):
        """Generate sample values in the hyperparameter search space of [0, 1]^k.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            samples (ArrayLike):
                2D arry with shape of (n_samples, self.K) with normalized values inside the search
                space [0, 1]^k.

        Examples:
            >>> instance = CategoricalHyperParam(choices=['Cat', 'Dog', 'Tiger'])
            >>> instance.sample(2)
            array([[1, 0, 0],
                   [0, 1, 0]])
        """
        randomized_values = np.random.random((n_samples, self.K))
        indexes = np.argmax(randomized_values, axis=1)

        sampled = [self.choices[index] for index in indexes]

        return self.transform(sampled)
