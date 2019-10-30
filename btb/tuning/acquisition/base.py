# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


class BaseAcquisition(metaclass=ABCMeta):

    def __init_acquisition__(self, **kwargs):
        pass

    @staticmethod
    def _get_max_candidates(candidates, n):
        sorted_candidates = list(reversed(np.argsort(candidates)))
        return sorted_candidates[:n]

    @abstractmethod
    def _acquire(self, candidates, num_candidates=1):
        """Decide which candidates to return as proposals.

        Apply a decision function to select the best candidates from
        the predicted scores list.

        Once the best candidates are found, their indexes are returned.

        Args:
            candidates (numpy.ndarray):
                2D array with two columns: scores and standard deviations
            num_candidates (int):
                Number of candidates to return.

        Returns:
            numpy.ndarray:
                Selected candidates indexes.
        """
        pass
