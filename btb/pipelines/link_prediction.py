import numpy as np
from btb.hyper_parameter import HyperParameter
from .dm_pipeline import DmPipeline

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import networkx as nx

class LinkPredictionPipeline(DmPipeline):

    HYPERPARAMETER_RANGES = [
        ('n_estimators', HyperParameter('int', [10, 500])),
        ('max_depth', HyperParameter('int', [3, 20]))
    ]


    D3M_PRIMITIVES = ["sklearn.preprocessing.LabelEncoder",
                      "sklearn.ensemble.RandomForestClassifier"]


    def cv_score(self, d3mds, cv_scoring = None, cv = None):
        """splits and scores and returns cv score"""
        # todo use scoring from problem
        cv_score = self._build_model(d3mds, cv = 3, cv_scoring = None)
        return cv_score

    def fit(self, d3mds):
        self.train_target_name = d3mds.problem.get_targets()[0]["colName"]
        self.model = self._build_model(d3mds)
        return self

    def predict(self, d3mds):
        """Make predictions on fitted pipeline"""
        X = self._featurize(d3mds, self.link_nodes, self.link_graph_res_id)
        X = self._impute(X)
        X = self._encode(X, self.encoded_values)
        out_predict = self.model.predict(X)
        out_predict = self.target_encoder.inverse_transform(out_predict)

        out_df = pd.DataFrame()
        out_df["d3mIndex"] = X.index.values
        out_df[self.train_target_name] = out_predict
        return out_df

    def _build_model(self, d3mds, cv_scoring = None, cv = None):
        y = pd.Series(d3mds.get_train_targets()[:, 0])
        X = self._featurize(d3mds)

        # handle missing values in an easy way
        # NOTE: this must be repeated in executable
        print("\n\nHandling missing values")
        X = self._impute(X)

        # handle categorical
        self.encoded_values = {}
        for c in X.columns:
            # todo make more robust detection of categorical
            if not np.issubdtype(X[c].dtype, np.number):
                counts = X[c].value_counts()
                self.encoded_values[c] = counts.head(10).index

        X = self._encode(X, self.encoded_values)

        print(("Training on %d features" % X.shape[1]))

        self.recommended_scoring = 'roc_auc'
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(y)
        model = RandomForestClassifier(**self.hyperparams, n_jobs=-1, verbose=False)

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



    def _featurize(self, d3mds, link_nodes=None, link_graph_res_id=None):
        "Fits model to given dataset for given hyperparameters"""
        df = d3mds.get_data_all(dropTargets=True)
        graphs = d3mds.dataset.get_graphs_as_nx()

        if link_nodes is None:
            # find the two nodes that we are predicting on
            self.link_nodes = []
            columns = d3mds.dataset.get_learning_data_columns()
            self.link_graph_res_id = None
            for c in columns:
                refers_to = c.get("refersTo", None)
                if refers_to is not None and refers_to["resID"] in graphs:
                    assert refers_to["resObject"] == "node"
                    self.link_nodes.append(c["colName"])

                    self.link_graph_res_id = refers_to["resID"]

                    # only can handle two link nodes
                    if len(self.link_nodes) == 2:
                        break

            link_nodes = self.link_nodes
            link_graph_res_id = self.link_graph_res_id

        G = nx.Graph(graphs[link_graph_res_id])
        pairs = df[link_nodes].values

        jc = nx.jaccard_coefficient(G, pairs)
        rai = nx.resource_allocation_index(G, pairs)
        aai = nx.adamic_adar_index(G, pairs)
        pa = nx.preferential_attachment(G, pairs)

        all_measures = [("jc",jc), ("rai",rai), ("aai", aai), ("pa", pa)]

        # make sure to use the index that we loaded
        fm = pd.DataFrame(index=df.index)
        for name, vals in all_measures:
            vals = np.array(list(vals))
            fm[name] = vals[:, 2]

        return fm

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



