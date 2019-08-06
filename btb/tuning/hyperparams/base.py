# -*- coding: utf-8 -*-

"""Package where the BaseHyperparameter is defined."""

from abc import ABCMeta, abstractmethod


class BaseHyperParam(metaclass=ABCMeta):
    """Base hyperparameter class.

    Abstract representation of a single hyperparameter that needs to be tuned.

    Attributes:
        K (int):
            Number of dimensions that this HyperParam uses to be represented in the search space.
    """

    @abstractmethod
    def inverse_transform(self, values):
        """Inverse transform one or more hyperparameter values.

        Transform one or more hyperparameter values from the normalized search
        space [0, 1]^k to the original hyperparameter space.

        Args:
            values (ArrayLike):
                Single value or 2D ArrayLike of normalized values.

        Returns:
            denormalized (Union[object, List[object]]):
                Denormalized value or list of denormalized values.
        """
        pass

    @abstractmethod
    def sample(self, n_samples):
        """Generate an array of ``num_samples`` random samples in the search space.

        Args:
            n_samples (int):
                Number of values to sample.

        Returns:
            samples (ArrayLike):
                2D array with of shape (n_samples, self.K)
        """
        pass

    @abstractmethod
    def transform(self, values):
        """Transform one or more hyperparameter values.

        Transform one or more hyperparameter values from the original hyperparameter space to the
        normalized search space [0, 1]^k.

        Args:
            values (Union[object, List[object]]):
                Single value or list of values to normalize.

        Returns:
            normalized (ArrayLike):
                2D array of shape(len(values), self.K)
        """
        pass
