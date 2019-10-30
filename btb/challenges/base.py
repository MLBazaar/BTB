# -*- coding: utf-8 -*-

"""Package where the Challenge class is defined."""

from abc import ABCMeta, abstractmethod


class Challenge(metaclass=ABCMeta):
    """Challenge class."""

    @abstractmethod
    def get_tunable(self):
        """Creates all the hyperparameters needed and instance a Tunable with them.

        Creates all the hyperparameters to be used for tuning and then instantitate a ``Tunable``
        object that will be returned for the ``Tuner``.

        Returns:
            ``btb.tuning.Tunable``:
                A ``Tunable`` instance containing a collection of hyperparameters to be tuned.
        """
        pass

    @abstractmethod
    def score(self):
        pass
