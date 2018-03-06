import numpy as np
from scipy import sparse
from btb.hyper_parameter import HyperParameter, ParamTypes
from .dm_pipeline import DmPipeline

import pandas as pd
from lightfm import LightFM

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


SCORING_FUNCS = {
    'mae': mean_absolute_error,
}

# TODO: include any extra columns in fm as user_features


class CollaborativeFilteringPipeline(DmPipeline):
    """Pipeline class for collaborative filtering problems

    Class variables:
        HYPERPARAMETER_RANGES: The default hyperparam_ranges.
            List of HyperParameter objects

    Attributes:
        hyperparam_ranges: List of HyperParameter objects, can be fine tuned from
            class defauls based on cpus/ram/dataset.
        d3m_primitives: list of strings of d3m primitives used in pipeline
        recommended_scoring: string represengint suggested scoring metric
        d3mds: object of D3M dataset for pipeline
        cpus: cpus information of system that pipeline is run on. Used to fine
            tune hyperparam ranges
        ram: ram of system that pipeline is run on. Used to fine tune hyperparam
            ranges.

    """

    HYPERPARAMETER_RANGES = [
        # TODO: ask about categoricals/strings
        ('loss', HyperParameter(ParamTypes.INT, [0, 3])),
        ('k', HyperParameter(ParamTypes.INT, [2, 10])),
        ('learning_schedule', HyperParameter(ParamTypes.INT, [0, 1])),
        ('no_components', HyperParameter(ParamTypes.INT, [5, 15])),
        ('epochs', HyperParameter(ParamTypes.INT, [1, 5])),
        ('n_jobs', HyperParameter(ParamTypes.INT, [0, 0])),
    ]

    D3M_PRIMITIVES = []

    def __init__(self, cpus=None, ram=None):
        super(CollaborativeFilteringPipeline, self).__init__(cpus, ram)
        # Initialized in load_data
        self.train_target_name = None
        self.recommended_scoring = 'mae'

    def _load_problem_data(self, d3mds):
        self.train_target_name = d3mds.problem.get_targets()[0]["colName"]

    def _build_model(self, d3mds, cv_scoring=None, cv=3):
        fm = d3mds.get_train_data()
        train_targets = d3mds.get_train_targets()[:, 0]
        loss = ['warp', 'logistic', 'bpr', 'warp-kos'][self.hyperparams['loss']]
        learning_schedule = ['adagrad', 'adadelta'][self.hyperparams['learning_schedule']]
        self.model = LightFM(loss=loss,
                             no_components=self.hyperparams['no_components'],
                             k=self.hyperparams['k'],
                             learning_schedule=learning_schedule
                             )

        if cv:
            cv_scores = self._cv_score(fm.values, train_targets, self.model, cv, cv_scoring or self.recommended_scoring)

        X = sparse.csr_matrix((train_targets, (fm.iloc[:, 0], fm.iloc[:, 1])))
        self.model.fit(X,
                       epochs=self.hyperparams['epochs'],
                       num_threads=self.hyperparams['n_jobs'])
        return cv_scores

    def _cv_score(self, X, y, model, cv, scoring):
        print("Doing Cross Validation")

        splitter = KFold(n_splits=cv or 3)
        if isinstance(scoring, str):
            cv_scoring_name = scoring
            cv_scoring_func = SCORING_FUNCS[cv_scoring_name]
        else:
            cv_scoring_name, cv_scoring_func = scoring
        print(("Scoring: %s" % cv_scoring_name))
        cv_scores = []
        labels = None
        for train_index, test_index in splitter.split(X, y):
            X_t = X[train_index, :]
            X_v = X[test_index, :]
            y_t = y[train_index]
            y_v = y[test_index]

            X_t = sparse.csr_matrix((y_t, (X_t[:, 0], X_t[:, 1])))
            model.fit(X_t, epochs=self.hyperparams['epochs'],
                       num_threads=self.hyperparams['n_jobs'])
            predictions = model.predict(user_ids=X_v[:, 0],
                                        item_ids=X_v[:, 1],
                                        num_threads=self.hyperparams['n_jobs'])
            try:
                cv_scores.append(cv_scoring_func(y_v, predictions))
            except:
                if labels is None:
                    labels = y.unique().tolist()
                cv_scores.append(cv_scoring_func(y_v, predictions, labels=labels))
        cv_score = (np.mean(cv_scores), np.std(cv_scores))
        return cv_score


    def cv_score(self, d3mds, cv_scoring=None, cv=3):
        """Paramaters:
            cv_scoring: performance metric from problem, optional
            cv: int, cross-validation generator or an iterable, optional
        """
        # TODO: use performance metric from problem
        self._load_problem_data(d3mds)
        cv_score = self._build_model(d3mds, cv=cv, cv_scoring=cv_scoring)
        return cv_score

    def fit(self, d3mds):
        # fits model to given dataset for given hyperparameters
        print("Training model")
        self._load_problem_data(d3mds)
        self._build_model(d3mds)

    def predict(self, d3mds):
        """Make predictions on fitted pipeline"""
        fm = d3mds.get_data_all(dropTargets=True)
        X = fm.values
        predictions = self.model.predict(user_ids=X[:, 0],
                                         item_ids=X[:, 1],
                                         num_threads=self.hyperparams['n_jobs'])
        out_df = pd.DataFrame()
        out_df["d3mIndex"] = fm.index.values
        out_df[self.train_target_name] = predictions
        return out_df
