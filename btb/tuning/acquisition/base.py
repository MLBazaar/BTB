# -*- coding: utf-8 -*-

"""Package where the BaseAcquisitionFunction class is defined."""

from abc import ABCMeta, abstractmethod


class BaseAcquisitionFunction(metaclass=ABCMeta):

    @abstractmethod
    def _acquire(self, predictions, n_candidates):
        pass
