import numpy as np
import scipy.stats as stats
from sklearn.decomposition import NMF

from btb.recommendation.recommender import BaseRecommender
from btb.recommendation.uniform import UniformRecommender


class MFRecommender(BaseRecommender):
    """Matrix factorization recommender

    Uses Matrix Factorization to determine which pipeline to recommend.

    Args:
        n_components (int): Corresponds to the number of features to keep in matrix decomposition.
            Must be greater than the number of rows in matrix.
        r_minimum (int): The minimum number of past results this recommender needs in order to use
            Matrix Factorization for prediction. If not enough results are present during a
            ``predict``, a uniform recommender is used.
    """

    def __init__(self, dpp_matrix, n_components=100, r_minimum=5):
        super(MFRecommender, self).__init__(dpp_matrix)

        self.n_components = n_components
        self.r_minimum = r_minimum

        # Matrix Factorization model that reduces dimensionality from num pipelines space to
        # n_components space.
        self.mf_model = NMF(n_components=n_components, init='nndsvd')

        dpp_decomposed = self.mf_model.fit_transform(dpp_matrix)

        # Matrix of rankings for each row of dpp_matrix after matrix facorization has been applied.
        self.dpp_ranked = np.empty(dpp_decomposed.shape)
        for i in range(dpp_decomposed.shape[0]):
            rankings = stats.rankdata(
                dpp_decomposed[i, :],
                method='dense'
            )
            self.dpp_ranked[i, :] = rankings

        random_matching_index = np.random.randint(self.dpp_matrix.shape[0])

        # Row from dpp_matrix representing pipeline performances for the dataset that most closely
        # matches the new dataset D. Identified in fit.
        self.matching_dataset = self.dpp_matrix[random_matching_index, :]

    def fit(self, dpp_vector):
        """
        Finds row of self.dpp_matrix most closely corresponds to X by means
        of Kendall tau distance.
        https://en.wikipedia.org/wiki/Kendall_tau_distance

        Args:
            dpp_vector (np.array): Array with shape (n_components, )
        """

        # decompose X and generate the rankings of the elements in the
        # decomposed matrix
        dpp_vector_decomposed = self.mf_model.transform(dpp_vector)
        dpp_vector_ranked = stats.rankdata(
            dpp_vector_decomposed,
            method='dense',
        )

        max_agreement_index = None
        max_agreement = -1  # min value of Kendall Tau agremment
        for i in range(self.dpp_ranked.shape[0]):
            # calculate agreement between current row and X
            agreement, _ = stats.kendalltau(
                dpp_vector_ranked,
                self.dpp_ranked[i, :],
            )
            if agreement > max_agreement:
                max_agreement_index = i
                max_agreement = agreement

        if max_agreement_index is None:
            max_agreement_index = np.random.randint(self.dpp_matrix.shape[0])
        # store the row with the highest agreement for prediction
        self.matching_dataset = self.dpp_matrix[max_agreement_index, :]

    def predict(self, indices):
        num_tried_candidates = len(np.where(self.dpp_vector != 0)[0])
        if num_tried_candidates < self.r_minimum:
            return UniformRecommender(self.dpp_matrix).predict(indices)
        matching_scores = np.array(
            [self.matching_dataset[each] for each in indices]
        )
        return stats.rankdata(matching_scores, method='dense')
