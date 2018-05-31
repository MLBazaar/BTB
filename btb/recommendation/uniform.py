import logging

import numpy as np

from btb.recommendation.recommender import BaseRecommender

LOGGER = logging.getLogger('btb')


class UniformRecommender(BaseRecommender):
    """
    Uniform Recommender for recomending pipelines to try on a new dataset D
    based on performances of datasets on the different pipeline options.
    Recommends pipelines that would maximize the score value. Raondomly
    ranks pipelines.

    Attributes:
        dpp_matrix: (2D np.array) Dataset performance pipeline matrix. Num
            datasets by num pipelines matrix where each row i corresponds to a
            dataset and each collumn j corresponds to a pipeline. The
            dpp_marix[i,j] is the score of pipeline j on dataset i or 0 if
            pipeline j has not beentried on dataset i.
        dpp_vector: (1D np.array) Vector representing pipeline performances on
            a new dataset D.
    """

    def predict(self, indicies):
        """
        Predicts the relative rankings of the pipelines on dpp_vector for
        a series of pipelines represented by their indicies.

        Args:
            indicies: np.array of pipeline indicies, shape = (n_samples)

        Returns:
            y: np.array of predicted scores, shape = (n_samples)
        """
        return np.random.permutation(indicies.shape[0])
