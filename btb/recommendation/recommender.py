import numpy as np


class BaseRecommender(object):
    """Base recommender

    Base recommender class for recommending pipelines to try on a new dataset D based on
    performances of datasets on the different pipeline options. Recommends pipelines that would
    maximize the score value.

    Args:
        dpp_matrix (np.array): Sparse dataset performance matrix pertaining to pipeline scores on
            different dataset with shape ``(num_datasets, num_pipelines)``. Each row ``i``
            corresponds to a dataset and each column ``j`` corresponds to a pipeline. The element
            ``dpp_matrix[i,j]`` corresponds to the score of the pipeline ``j`` on the dataset and
            is ``0`` if the pipeline was not tried on the dataset.
    """

    def __init__(self, dpp_matrix):
        # Dataset performance pipeline matrix
        self.dpp_matrix = dpp_matrix

        # Vector representing pipeline performances on a new dataset D
        self.dpp_vector = np.zeros(self.dpp_matrix.shape[1])

    def fit(self, dpp_vector):
        """Fit the recommender model.

        Args:
            dpp_vector (np.array): Array with shape (n_components, )
        """
        self.dpp_vector = dpp_vector

    def predict(self, indices):
        """
        Predicts the relative rankings of the pipelines on dpp_vector for
        a series of pipelines represented by their indices.

        Args:
            indices (np.array): Array of pipeline indices with shape (n_samples)

        Returns:
            np.array: Array of predicted scores with shape (n_samples)
        """
        raise NotImplementedError

    def _acquire(self, predictions):
        """Finds the best candidate given a series of predictions.

        Args:
            predictions (np.array): Array of predictions corresponding to a set of possible
                pipelines.

        Returns:
            int: Index of the selected candidate from predictions
        """
        return np.argmax(predictions)

    def _get_candidates(self):
        """Finds the pipelines that are not yet tried.

        Returns:
            np.array: Indices corresponding to columns in ``dpp_matrix`` that haven't been tried on
                ``X``. ``None`` if all pipelines have been tried on X.
        """
        candidates = np.where(self.dpp_vector == 0)
        return None if len(candidates[0]) == 0 else candidates[0]

    def propose(self):
        """Use the trained model to propose a new pipeline.

        Returns:
            int: Index corresponding to pipeline to try in ``dpp_matrix``.
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
        """Add data about known pipeline and scores.

        Updates ``dpp_vector`` and refits model with all data.

        Args:
            X (dict): mapping of pipeline indices to scores. Keys must correspond to the index of a
                column in ``dpp_matrix`` and values are the corresponding score for pipeline on
                the dataset.
        """
        for each in X:
            self.dpp_vector[each] = X[each]
        self.fit(self.dpp_vector.reshape(1, -1))
