# -*- coding: utf-8 -*-


class Tunable:
    """Tunable class.

    Collection of HyperParams that need to be tuned as a whole.

    Attributes:
        hyperparams:
            List of hyperparameters from this hyperparam space.
    """

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def transform(self, values):
        """Transform one or more hyperparameter value combinations.

        Transform one or more hyperparameter value combinations from the original hyperparameter
        space the normalized search space [0, 1]^K.

        Args:
            values (ArrayLike): 2D array of shape (*, len(self.hyperparameters)).

        Returns:
            transformed (ArrayLike): 2D array of shape (len(values), K)
        """
        pass

    def inverse_transform(self, values):
        """Rever one or more hyperparameter value combinations.

        Transform one or more hyperparameter values from the normalized search space [0, 1]^K to
        the original hyperparameter space.

        Args:
            values (ArrayLike): 2D array of normalized values with shape (*, K).

        Returns:
            reversed (ArrayLike): 2D array of shape (len(values), len(self.hyperparameters)).
        """
        pass

    def sample(self, n_samples):
        """Sample values in this hyperparameter search space.

        Args:
            n_samlpes (int): Number of values to sample.

        Returns:
            samples (ArrayLike): 2D array with shape of (n_samples, self._k).
        """
        pass

    def to_dict(self):
        """Get a dict representation of this Tunable."""
        pass

    @classmethod
    def from_dict(cls, spec_dict):
        """Load a Tunable from a dict representation."""
        pass
