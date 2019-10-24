# -*- coding: utf-8 -*-

"""Package where the Challenge class is defined."""

from abc import ABCMeta, abstractmethod, abstractstaticmethod


class Challenge(metaclass=ABCMeta):
    """Challenge class.

    The Challenge class represents a single ``challenge`` that can be used for benchmark.
    """

    @abstractstaticmethod
    def get_tunable(cls):
        """Create the ``hyperparameters`` and return the ``tunable`` created with them.

        Returns:
            ``btb.tuning.Tunable``:
                A ``Tunable`` instance to be used to tune the ``self.score`` method.
        """
        pass

    @abstractmethod
    def score(self, *args, **kwargs):
        """Perform scoring with given ``arguments``.

        This method will score a result with a given configuration, then return the score obtained
        for those ``arguments``.
        """
        pass
