import numpy as np

from btb.recommendation.recommender import BaseRecommender


class UniformRecommender(BaseRecommender):
    """Uniform recommender

    Uniform Recommender for recomending pipelines to try on a new dataset D based on performances
    of datasets on the different pipeline options. Recommends pipelines that would maximize the
    score value. Randomly ranks pipelines.
    """

    def predict(self, indices):
        return np.random.permutation(indices.shape[0])
