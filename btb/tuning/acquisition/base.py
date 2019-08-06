# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class AcquisitionFunctionMixin(metaclass=ABCMeta):

    @abstractmethod
    def _acquire(self, predictions, n_candidates):
        pass
