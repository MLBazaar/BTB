import numpy as np
from btb.hyper_parameter import HyperParameter, ParamTypes
from .dm_pipeline import DmPipeline

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score


class SingleTablePipeline(DmPipeline):
    """Pipeline class for single table datasets

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
        ('n_estimators', HyperParameter('int', [10, 500])),
        ('max_depth', HyperParameter('int', [3, 20]))
    ]


    D3M_PRIMITIVES = ["sklearn.preprocessing.Imputer",
                      "sklearn.preprocessing.LabelEncoder",
                      "sklearn.ensemble.RandomForestClassifier",
                      "sklearn.ensemble.RandomForestRegressor"]

    def __init__(self, cpus=None, ram=None):
        super(SingleTablePipeline, self).__init__(cpus, ram)
        # Initialized in load_data
        self.task_type = None
        self.task_sub_type = None
        self.train_target_name = None


    def _load_problem_data(self, d3mds):
        self.task_type = d3mds.problem.get_taskType()
        self.task_sub_type = d3mds.problem.get_taskSubType()
        self.train_target_name = d3mds.problem.get_targets()[0]["colName"]

    def _build_model(self, d3mds, cv_scoring = None, cv = None):
        fm = d3mds.get_train_data()

        # handle missing values in an easy way
        # NOTE: this must be repeated in executable
        print("\n\nHandling missing values")
        fm = self._impute(fm)


        print("Merging features and targets")
        # todo don't assume fm and train targets are ordered
        # fm = fm.merge(train_targets, left_index=True, right_index=True)

        # prep for ML by separating labels and feature

        train_targets = pd.Series(d3mds.get_train_targets()[:, 0])
        y = train_targets
        X = fm

        # handle categorical
        self.encoded_values = {}
        for c in X.columns:
            # todo make more robust detection of categorical
            if not np.issubdtype(X[c].dtype, np.number):
                counts = X[c].value_counts()
                self.encoded_values[c] = counts.head(10).index

        X = self._encode(X, self.encoded_values)

        print(("Training on %d features" % X.shape[1]))

        # select model to use
        if self.task_type == "classification":
            if self.task_sub_type == "binary":
                print("Setting up binary classifier")
                self.recommended_scoring = 'roc_auc'
            elif self.task_sub_type == "multiClass":
                self.recommended_scoring = 'f1_micro'

                print("Setting up multiclass classifier")

            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)
            model = RandomForestClassifier(**self.hyperparams, n_jobs=-1, verbose=False)


        elif self.task_type == "regression":
            print("Setting up regressor")
            self.recommended_scoring = 'r2'
            model = RandomForestRegressor(**self.hyperparams, n_jobs=-1, verbose=False)

        else:
            raise Exception("Unsupported task type %s" % self.task_type)

        if cv:
            if cv_scoring is None:
                cv_scoring = self.recommended_scoring

            print("Doing Cross Validation")
            print(("Scoring: %s" % cv_scoring))
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=cv_scoring)
            cv_score = (cv_scores.mean(), cv_scores.std())
            return cv_score
        else:
            model.fit(X, y)
            return model

    def _encode(self, fm, encoded_values):
        fm_cols = fm.columns
        for c in fm_cols:
            if c in encoded_values:
                # remove existing col
                col = fm.pop(c)
                for val in encoded_values[c]:
                    encoded_name = "%s=%s" % (str(c), str(val))
                    fm[encoded_name] = col == val
        return fm

    def _impute(self, fm):
        return fm.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

    def cv_score(self, d3mds, cv_scoring = None, cv = None):
        """Paramaters:
            cv_scoring: performance metric from problem, optional
            cv: int, cross-validation generator or an iterable, optional
        """

        # TODO: use performance metric from problem
        self._load_problem_data(d3mds)
        cv_score = self._build_model(d3mds, cv = 3, cv_scoring = None,)
        return cv_score

    def fit(self, d3mds):
        # fits model to given dataset for given hyperparameters
        print("Training model")
        self._load_problem_data(d3mds)

        self.model = self._build_model(d3mds)

    def predict(self, d3mds):
        """Make predictions on fitted pipeline"""
        fm = d3mds.get_data_all(dropTargets=True)

        # run fitted pipeline
        fm = self._impute(fm)
        fm = self._encode(fm, self.encoded_values)
        out_predict = self.model.predict(fm)

        # prepare output
        if self.task_sub_type in ['multiClass', 'binary']:
            out_predict = self.target_encoder.inverse_transform(out_predict)

        out_df = pd.DataFrame()
        out_df["d3mIndex"] = fm.index.values
        out_df[self.train_target_name] = out_predict
        return out_df
