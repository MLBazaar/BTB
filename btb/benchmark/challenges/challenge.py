# -*- coding: utf-8 -*-

"""Package where the Challenge class is defined."""

import inspect
from abc import ABCMeta, abstractmethod


class Challenge(metaclass=ABCMeta):
    """Challenge class.

    The Challenge class represents a single ``challenge`` that can be used for benchmark.
    """

    @abstractmethod
    def get_tunable_hyperparameters(self):
        """Create the ``hyperparameters`` and return the ``tunable`` created with them.

        Returns:
            ``btb.tuning.Tunable``:
                A ``Tunable`` instance to be used to tune the ``self.score`` method.
        """
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Perform evaluation for the given ``arguments``.

        This method will score a result with a given configuration, then return the score obtained
        for those ``arguments``.
        """
        pass

    def __repr__(self):
        args = ', '.join(
            '{}={}'.format(param, getattr(self, param))
            for param in inspect.signature(self.__class__).parameters.keys()
        )
        return '{}({})'.format(self.__class__.__name__, args)


class MLChallenge(Challenge):
    """Machine Learning Challenge class.

    The MLChallenge class rerpresents a single ``machine learning challenge`` that can be used for
    benchmark.
    """

    @abstractmethod
    def _load_data(self):
        """Load ``X`` and ``y`` over which to perform fit and evaluate."""
        pass
