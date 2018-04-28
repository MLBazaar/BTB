import logging

import numpy as np
import scipy.stats as stats

from sklearn.decomposition import NMF

logger = logging.getLogger('btb')


class Recommender(object):
    """
    Basic Recommender for recomending pipelines to try on a new dataset D based
    on performances of datasets on the different pipeline options. Recommends
    pipelines that would maximize the score value.

    Attributes:
        dpp_matrix: (2D np.array) Dataset performance pipeline matrix. Num
            datasets by num pipelines matrix where each row i corresponds to a
            dataset and each collumn j corresponds to a pipeline. The
            dpp_marix[i,j] is the score of pipeline j on dataset i or 0 if
            pipeline j has not beentried on dataset i.
        n_components: (int) The number of components (columsn) to keep after
            Matrix Factorxation.
        mf_model: Matrix Factorization model that reduces dimensionailty from
            num pipelines space to n_components space.
        dpp_ranked: (2D np.array) Matrix of rankings for each row of dpp_matrix
            after matrix facorization has been applied.
        dpp_vector: (1D np.array) Vector representing pipeline performances on
            a new dataset D.
        matching_dataset: (1D np.array) Row from dpp_matrix representing
            pipeline performances for the dataset that most closely matches the
            new dataset D. Identified in fit.
    """

    def __init__(self, dpp_matrix, n_components=100, **kwargs):
        """
        Args:
            dpp_matrix: np.array shape = (num_datasets, num_pipelines) Sparse
                dataset performance matrix pertaining to pipeline
                scores on different dataset. Each row i coresponds to a dataset
                and each column j corresponds to a pipeline. dpp_matrix[i,j]
                corresponds to the score of the pipeline j on the dataset and
                is 0 if the pipeline was not tried on the dataset
            n_components: int. Corresponds to the number of features to keep
                in matrix decomposition. Must be >= number of rows in matrix
        """
        self.dpp_matrix = dpp_matrix
        self.n_components = n_components
        self.mf_model = NMF(
            n_components=n_components,
            init='nndsvd',
        )
        dpp_decomposed = self.mf_model.fit_transform(dpp_matrix)
        self.dpp_ranked = np.empty(dpp_decomposed.shape)
        for i in range(dpp_decomposed.shape[0]):
            rankings = stats.rankdata(
                dpp_decomposed[i, :],
                method='dense'
            )
            self.dpp_ranked[i, :] = rankings
        self.dpp_vector = np.zeros(self.dpp_matrix.shape[1])
        self.matching_dataset = None

    def fit(self, dpp_vector):
        """
        Finds row of self.dpp_matrix most closely corresponds to X by means
        of Kendall tau distance.
        https://en.wikipedia.org/wiki/Kendall_tau_distance

        Args:
            dpp_vector: np.array shape = (self.n_components,)
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
            agreement, p_value = stats.kendalltau(
                dpp_vector_ranked,
                self.dpp_ranked[i, :],
            )
            if agreement > max_agreement:
                max_agreement_index = i
                max_agreement = agreement
        # store the row with the highest agreement for prediction
        self.matching_dataset = self.dpp_matrix[max_agreement_index, :]

    def predict(self, indicies):
        """
        Predicts the relative rankings of the pipelines on dpp_vector for
        a series of pipelines represented by their indicies.

        Args:
            indicies: np.array of pipeline indicies, shape = (n_samples)

        Returns:
            y: np.array of predicted scores, shape = (n_samples)
        """
        matching_scores = np.array(
            [self.matching_dataset[each] for each in indicies]
        )
        return stats.rankdata(matching_scores, method='dense')

    def _acquire(self, predictions):
        """
        Acquisition function. Finds the best candidate given a series of
        predictions.

        Args:
            predictions: np.array of predictions, corresponding to a set of
                possible pipelines.

        Returns:
            idx: index of the selected candidate from predictions
        """
        return np.argmax(predictions)

    def _get_candidates(self):
        """
        Finds the pipelines that are not yet tried for the new dataset D
        represented by dpp_vector.

        Returns:
            indicies: np.array. Indicies corresponding to collumns in
                self.dpp_matrix that haven't been tried on X.
                None if all pipelines have been tried on X.
        """
        candidates = np.where(self.dpp_vector == 0)
        return None if len(candidates[0]) == 0 else candidates[0]

    def propose(self):
        """
        Use the trained model to propose a new pipeline.

        Returns:
            proposal: int. index corresponding to pipeline to try. the
                pipeline corresponds to the proposal'th column of
                self.dpp_matrix
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
        Updates self.dpp_vector.
        Adds data about known pipelines and scores.
        Refits model with all data.
        Args:
            X: dict mapping pipeline indicies to scores.
                Keys must correspond to the index of a collumn in
                self.dpp_matrix
                Values are the corresponding score for pipeline on the dataset
        """
        for each in X:
            self.dpp_vector[each] = X[each]
        self.fit(self.dpp_vector.reshape(1, -1))
