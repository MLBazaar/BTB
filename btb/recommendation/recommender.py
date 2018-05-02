import logging

import numpy as np

logger = logging.getLogger('btb')


class BaseRecommender(object):
    """
    Base Recommender class for recomending pipelines to try on a new dataset
    D based on performances of datasets on the different pipeline options.
    Recommends pipelines that would maximize the score value.

    Attributes:
        dpp_matrix: (2D np.array) Dataset performance pipeline matrix. Num
            datasets by num pipelines matrix where each row i corresponds to a
            dataset and each collumn j corresponds to a pipeline. The
            dpp_marix[i,j] is the score of pipeline j on dataset i or 0 if
            pipeline j has not beentried on dataset i.
        dpp_vector: (1D np.array) Vector representing pipeline performances on
            a new dataset D.
    """

    def __init__(self, dpp_matrix, **kwargs):
        """
        Args:
            dpp_matrix: np.array shape = (num_datasets, num_pipelines) Sparse
                dataset performance matrix pertaining to pipeline
                scores on different dataset. Each row i coresponds to a dataset
                and each column j corresponds to a pipeline. dpp_matrix[i,j]
                corresponds to the score of the pipeline j on the dataset and
                is 0 if the pipeline was not tried on the dataset
        """
        self.dpp_matrix = dpp_matrix
        self.dpp_vector = np.zeros(self.dpp_matrix.shape[1])

    def fit(self, dpp_vector):
        """
        Fits the Recommender model.
        Args:
            dpp_vector: np.array shape = (self.n_components,)
        """
        self.dpp_vector = dpp_vector

    def predict(self, indicies):
        """
        Predicts the relative rankings of the pipelines on dpp_vector for
        a series of pipelines represented by their indicies.

        Args:
            indicies: np.array of pipeline indicies, shape = (n_samples)

        Returns:
            y: np.array of predicted scores, shape = (n_samples)
        """
        raise NotImplementedError(
            'predict() needs to be implemented by a' +
            'subclass of BaseRecommender'
        )

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
