# -*- coding: utf-8 -*-

"""Package where the Challenge class is defined."""

from abc import ABCMeta, abstractmethod


class Challenge(metaclass=ABCMeta):
    """Challenge class.

    The Challenge class represents a single ``challenge`` that can be used for benchmark.
    """

    @abstractmethod
    def get_tunable(self):
        """Create the ``hyperparameters`` and return the ``tunable`` created with them.

        Returns:
            ``btb.tuning.Tunable``:
                A ``Tunable`` instance to be used to tune the ``self.score`` method.
        """

    def get_tuner_params(self):
        """Obtain the configuration needed for the ``Tuner`` to work with this challenge.

        Return:
            dict:
                A dictionary containing the needed parameters for the ``Tuner`` to work properly
                with this challenge.
        """
        return {}

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Perform evaluation for the given ``arguments``.

        This method will score a result with a given configuration, then return the score obtained
        for those ``arguments``.
        """
