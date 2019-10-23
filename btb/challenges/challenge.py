# -*- coding: utf-8 -*-

"""Package where the Challenge class is defined."""

from abc import ABCMeta, abstractclassmethod, abstractmethod


class Challenge(metaclass=ABCMeta):
    """Challenge class.

    A Challenge is an abstract representation of a single ``challenge``. This class
    is ment to be used for benchmark.
    """

    @abstractclassmethod
    def get_tunable(cls):
        """Create the hyperparameters and return the tunable created with them.

        Returns:
            ``btb.tuning.Tunable``:
                A ``Tunable`` instance to be used to tune the ``self.score`` method.
        """
        pass

    @abstractmethod
    def score(self, *args, **kwargs):
        """Return a score."""
        pass
