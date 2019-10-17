# -*- coding: utf-8 -*-

"""Package where the BaseAcquisitionFunction class is defined."""

from abc import ABCMeta, abstractmethod


class BaseAcquisitionFunction(metaclass=ABCMeta):

    def __init_acquisition__(self):
        """Define the requiered initiation that the acquisition may requiere."""
        pass

    @abstractmethod
    def _acquire(self, candidates, num_candidates):
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
