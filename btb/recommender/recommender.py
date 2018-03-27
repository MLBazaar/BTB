import logging
from builtins import range, object
import numpy as np
import random
import math
from sklearn.decomposition import NMF
import scipy.stats as stats


logger = logging.getLogger('btb')


class Recommender(object):
    def __init__(self, matrix, n_components=100, **kwargs):
        """
        Args:
            matrix: np.array Sparse matrix pertaining to pipeline scores on
                different dataset. Each row i coresponds to a dataset and each
                column j corresponds to a pipeline. matrix[i,j] corresponds
                to the score of the pipeline j on the dataset adn is 0 if the
                pipeline was not tried on the dataset
            n_components: int. Corresponds to the number of features to keep
                in matrix decomposition. Must be >= number of rows in matrix
        """
        self.matrix = matrix
        self.n_components = n_components
        self.mf_model = NMF(
            n_components=n_components,
            init='nndsvd',
        )
        decomposition_matrix = self.mf_model.fit_transform(matrix)
        self.ranking_matrix = np.empty(decomposition_matrix.shape)
        for i in range(decomposition_matrix.shape[0]):
            rankings = stats.rankdata(
                decomposition_matrix[i, :],
                method='dense'
            )
            self.ranking_matrix[i, :] = rankings
        self.X = np.zeros(self.matrix.shape[1])
        self.closest_row = None

    def fit(self, X):
        """
        Fits the model for a new dataset & pipeline score entry X
        Finds row of self.matrix most closely corresponds to X by means
        of Kendall tau distance.
        https://en.wikipedia.org/wiki/Kendall_tau_distance

        Args:
            X: np.array shape = (self.n_components,)
        """

        # decompose X and generate the rankings of the elements in the
        # decomposed matrix
        x_decomposed = self.mf_model.transform(X)
        x_rankings = stats.rankdata(x_decomposed, method='dense')

        max_agrement_index = None
        max_agreement = -1  # min value of Kendall Tau agremment
        for i in range(self.ranking_matrix.shape[0]):
            # calculate agreement between current row and X
            agreement, p_value = stats.kendalltau(
                x_rankings,
                self.ranking_matrix[i, :],
            )
            if agreement > max_agreement:
                max_agrement_index = i
                max_agreement = agreement

        # store the row with the highest agreement for prediction
        self.closest_row = self.matrix[i, :]

    def predict(self, indicies):
        """
        Predicts the pipeline score on X for a series of pipelines represented
        by their indicies.

        Args:
            indicies: np.array of pipeline indicies, shape = (n_samples)

        Returns:
            y: np.array of predicted scores, shape = (n_samples)
        """
        return np.array([self.closest_row[each] for each in indicies])

    def _acquire(self, predictions):
        """
        Acquisition function. Accepts a li.reshape(1, -1)f the best candidate.

        Args:
            predictions: np.array of predictions, corresponding to a set of
                possible pipelines.

        Returns:
            idx: index of the selected candidate from predictions
        """
        return np.argmax(predictions)

    def _get_candidates(self):
        """
        Gets the list of untried candidate pipelines on X.

        Returns:
            indicies: np.array. Indicies corresponding to collumns in
                self.matrix that haven't been tried on X.
                None if all pipelines have been tried on X.
        """
        candidates = np.where(self.X == 0)
        return None if len(candidates[0]) == 0 else candidates[0]

    def propose(self):
        """
        Use the trained model to propose a new pipeline.

        Returns:
            proposal: int. index corresponding to pipeline to try. the
                pipeline corresponds to the proposal'th column of self.matrix
        """
        # generate a list of all the untried candidate pipelines
        candidates = self._get_candidates()

        # get_candidates() returns None when every possibility has been tried
        if candidates is None:
            return None

        # predict() returns a predicted values for each candidate
        predictions = self.predict(candidates)

        # acquire() evaluates the list of predictions, selects one, and returns
        # its index.
        idx = self._acquire(predictions)
        return candidates[idx]

    def add(self, X):
        """
        Adds data about known pipelines and scores.
        Refits model with all data.
        Args:
            X: dict mapping pipeline indicies to scores.
                Keys must correspond to the index of a collumn in self.matrix
                Values are the corresponding score for pipeline on the dataset
        """
        for each in X:
            self.X[each] = X[each]
        self.fit(self.X.reshape(1, -1))
