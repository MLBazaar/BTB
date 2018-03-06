import numpy as np
import copy
from btb.hyper_parameter import HyperParameter
from .dm_pipeline import DmPipeline

import pandas as pd
from d3m.primitives.sklearn_wrap import (SKRandomForestClassifier,
                                         SKRandomForestRegressor,
                                         SKStandardScaler)
from d3m.primitives.featuretools_ta1 import (DFS, Imputer,
                                             RFRegressorFeatureSelector,
                                             RFClassifierFeatureSelector)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, r2_score


def f1_micro_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


SCORING_FUNCS = {
    'f1': f1_score,
    'r2': r2_score,
    'f1_micro': f1_micro_score,
}


class MultitablePipeline(DmPipeline):
    """Pipeline class for single or multi table datasets

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
        ('selector_n_estimators', HyperParameter('int', [10, 500])),
        ('modeler_n_estimators', HyperParameter('int', [10, 500])),
        ('select_n_features', HyperParameter('int', [10, 500])),
        ('dfs_max_depth', HyperParameter('int', [1, 3])),
    ]


    D3M_PRIMITIVES = ["d3m.primitives.sklearn_wrap.SKRandomForestRegressor",
                      "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
                      "d3m.primitives.sklearn_wrap.SKStandardScaler",
                      "d3m.primitives.featuretools_ta1.Imputer",
                      "d3m.primitives.featuretools_ta1.RFClassifierFeatureSelector",
                      "d3m.primitives.featuretools_ta1.RFRegressorFeatureSelector",
                      "d3m.primitives.featuretools_ta1.DFS"]

    def __init__(self, cpus=None, ram=None):
        super(MultitablePipeline, self).__init__(cpus, ram)
        # Initialized in load_data
        self.task_type = None
        self.task_sub_type = None
        self.train_target_name = None

    def _load_problem_data(self, d3mds):
        self.task_type = d3mds.problem.get_taskType()
        self.task_sub_type = d3mds.problem.get_taskSubType()
        self.train_target_name = d3mds.problem.get_targets()[0]["colName"]

    def _build_model(self, d3mds, cv_scoring=None, cv=3):
        target = {'column_name': self.train_target_name}

        metadata = DFS.metadata.query()['primitive_code']
        DFSHP = metadata['class_type_arguments']['Hyperparams']
        dfs_hp = DFSHP(DFSHP.defaults(),
                       max_depth=self.hyperparams['dfs_max_depth'])
        self.dfs = DFS(hyperparams=dfs_hp)
        self.dfs.set_training_data(inputs=[d3mds, target])
        self.dfs.fit()
        fm, encoded_features = self.dfs.produce_encoded(inputs=[d3mds, target]).value
        fm.reset_index('time', drop=True, inplace=True)
        self.dfs._features = encoded_features

        train_targets = d3mds.get_train_targets()[:, 0]
        y = train_targets
        self.train_y = y

        feature_names = fm.columns

        metadata = Imputer.metadata.query()['primitive_code']
        ImputerHP = metadata['class_type_arguments']['Hyperparams']
        self.imputer = Imputer(hyperparams=ImputerHP.defaults())
        fm = self.imputer.produce(inputs=fm).value

        metadata = SKStandardScaler.metadata.query()['primitive_code']
        ScalerHP = metadata['class_type_arguments']['Hyperparams']
        self.scaler = SKStandardScaler(hyperparams=ScalerHP.defaults())
        self.scaler.set_training_data(inputs=fm.values)
        self.scaler.fit()
        fm = self.scaler.produce(inputs=fm.values).value

        fm = pd.DataFrame(fm, columns=feature_names)
        X = fm

        # select model to use
        if self.task_type == "classification":
            if self.task_sub_type == "binary":
                print("Setting up binary classifier")
                self.recommended_scoring = ('f1', f1_score)
            elif self.task_sub_type == "multiClass":
                self.recommended_scoring = ('f1_micro', f1_micro_score)

                print("Setting up multiclass classifier")

            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)
            X, self.selector = self.select_features_classification(X, y)
            self.model, cv_model = self.train_classifier(X, y)
            splitter = StratifiedKFold(n_splits=cv or 3)
        elif self.task_type == "regression":
            print("Setting up regressor")
            self.recommended_scoring = ('r2', r2_score)
            X, self.selector = self.select_features_regression(X, y)
            self.model, cv_model = self.train_regressor(X, y)
            splitter = KFold(n_splits=cv or 3)
        else:
            raise Exception("Unsupported task type %s" % self.task_type)

        if cv:
            cv_scores = self._cv_score(X, y, cv_model, splitter, cv_scoring or self.recommended_scoring)
            return cv_scores

    def select_features_regression(self, X, y):
        metadata = RFRegressorFeatureSelector.metadata.query()['primitive_code']
        SelectorHP = metadata['class_type_arguments']['Hyperparams']

        selector = None
        if self.hyperparams['select_n_features'] < X.shape[1]:
            selector_hp = SelectorHP(SelectorHP.defaults(),
                                     select_n_features=self.hyperparams['select_n_features'],
                                     n_estimators=self.hyperparams['selector_n_estimators'],
                                     n_jobs=-1)
            selector = RFRegressorFeatureSelector(hyperparams=selector_hp)
            selector.set_training_data(inputs=X,
                                       outputs=y)
            selector.fit()
            X = selector.produce(inputs=X).value
        return X, selector

    def select_features_classification(self, X, y):
        metadata = RFClassifierFeatureSelector.metadata.query()['primitive_code']
        SelectorHP = metadata['class_type_arguments']['Hyperparams']

        selector = None
        if self.hyperparams['select_n_features'] < X.shape[1]:
            selector_hp = SelectorHP(SelectorHP.defaults(),
                                     select_n_features=self.hyperparams['select_n_features'],
                                     n_estimators=self.hyperparams['selector_n_estimators'],
                                     n_jobs=-1)
            selector = RFClassifierFeatureSelector(hyperparams=selector_hp)
            # JPL's wrapper doesn't allow strings for max_features
            selector._selector._clf.max_features = 'auto'
            selector.set_training_data(inputs=X,
                                       outputs=y)
            selector.fit()
            X = selector.produce(inputs=X).value
        return X, selector

    def train_regressor(self, X, y):
        metadata = SKRandomForestRegressor.metadata.query()['primitive_code']
        RFHyperparams = metadata['class_type_arguments']['Hyperparams']

        hp = RFHyperparams(RFHyperparams.defaults(),
                           n_estimators=self.hyperparams['modeler_n_estimators'],
                           n_jobs=-1,
                           max_features='auto')
        rf = SKRandomForestRegressor(hyperparams=hp)

        cv_rf = copy.deepcopy(rf)
        rf.set_training_data(inputs=X, outputs=y)
        rf.fit()
        return rf, cv_rf


    def train_classifier(self, X, y):
        metadata = SKRandomForestClassifier.metadata.query()['primitive_code']
        RFHyperparams = metadata['class_type_arguments']['Hyperparams']

        hp = RFHyperparams(RFHyperparams.defaults(),
                           n_estimators=self.hyperparams['modeler_n_estimators'],
                           n_jobs=-1)
        rf = SKRandomForestClassifier(hyperparams=hp)
        # JPL's wrapper doesn't allow strings for max_features
        rf._clf.max_features = 'auto'

        cv_rf = copy.deepcopy(rf)
        rf.set_training_data(inputs=X, outputs=y)
        rf.fit()
        return rf, cv_rf

    def _cv_score(self, X, y, model, splitter, scoring):
        print("Doing Cross Validation")
        if isinstance(scoring, str):
            cv_scoring_name = scoring
            cv_scoring_func = SCORING_FUNCS[cv_scoring_name]
        else:
            cv_scoring_name, cv_scoring_func = scoring
        print(("Scoring: %s" % cv_scoring_name))
        cv_scores = []
        labels = None
        for train_index, test_index in splitter.split(X, y):
            X_t = X.iloc[train_index]
            X_v = X.iloc[test_index]
            y_t = y[train_index]
            y_v = y[test_index]
            model.set_training_data(inputs=X_t, outputs=y_t)
            model.fit()
            predictions = model.produce(inputs=X_v).value
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
        target = {'column_name': self.train_target_name}
        fm = self.dfs.produce(inputs=[d3mds, target]).value
        feature_names = fm.columns
        fm = self.imputer.produce(inputs=fm).value
        fm = self.scaler.produce(inputs=fm.values).value
        fm = pd.DataFrame(fm, columns=feature_names)
        if self.selector is not None:
            fm = self.selector.produce(inputs=fm).value
        out_predict = self.model.produce(inputs=fm).value

        # prepare output
        if self.task_sub_type in ['multiClass', 'binary']:
            out_predict = self.target_encoder.inverse_transform(out_predict)

        out_df = pd.DataFrame()
        out_df["d3mIndex"] = fm.index.values
        out_df[self.train_target_name] = out_predict
        return out_df

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        if self.model is not None:
            model = d['model']
            del d['model']
            d['model_params'] = model.get_params()
            d['model_hyperparams'] = model.hyperparams
            d['model_class'] = type(model)
        return d

    def __setstate__(self, d):
        if 'model_class' in d:
            self.model = d['model_class'](hyperparams=d['model_hyperparams'])
            self.model.set_params(params=d['model_params'])
            del d['model_class']
            del d['model_hyperparams']
            del d['model_params']
        for k, v in d.items():
            setattr(self, k, v)
