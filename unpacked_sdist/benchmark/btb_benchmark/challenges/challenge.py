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
        """Return a dictionary with hyperparameters to be tuned."""
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Perform evaluation for the given ``arguments``.

        This method will score a result with a given configuration, then return the score obtained
        for those ``arguments``.
        """
        pass

    def __repr__(self):
        args = inspect.getargspec(self.__init__)
        keys = args.args[1:]
        defaults = dict(zip(keys, args.defaults))
        instanced = {key: getattr(self, key) for key in keys}

        if defaults == instanced:
            return '{}()'.format(self.__class__.__name__)

        else:
            args = ', '.join(
                '{}={}'.format(key, value)
                for key, value in instanced.items()
            )

            return '{}({})'.format(self.__class__.__name__, args)
