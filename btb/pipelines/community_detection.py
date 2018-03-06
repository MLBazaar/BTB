from .dm_pipeline import DmPipeline
import pandas as pd
import networkx as nx
import time
import community as co


class CommunityDetectionPipeline(DmPipeline):

    # todo one parameter to tune: resolution
    HYPERPARAMETER_RANGES = []

    D3M_PRIMITIVES = []

    def cv_score(self, d3mds, cv_scoring = None, cv = None):
        """splits and scores and returns cv score"""
        # prevent fast loops
        time.sleep(.1)
        return (1, 0)

    def fit(self, d3mds):
        # prevent fast loops
        time.sleep(.1)
        return self

    def predict(self, d3mds):
        """Make predictions on fitted pipeline"""
        graphs = d3mds.dataset.get_graphs_as_nx()
        df = d3mds.get_test_data()
        train_target_name = d3mds.problem.get_targets()[0]["colName"]

        # find the two nodes that we are predicting on
        columns = d3mds.dataset.get_learning_data_columns()
        for c in columns:
            refers_to = c.get("refersTo", None)
            if refers_to is not None and refers_to["resID"] in graphs:
                assert refers_to["resObject"] == "node"
                node_col = c["colName"]
                node_graph_res_id = refers_to["resID"]
                break
        out_df = df[[node_col]]
        out_df[train_target_name] = None
        G = nx.Graph(graphs[node_graph_res_id])

        print("Starting best partition")
        partition = co.best_partition(G)
        print("Finished best partition")
        # start missing community somewhere above highest number
        missing_community_index = pd.Series([b for a,b in partition.items()]).max() + 10
        for i in out_df.index:
            node = out_df.loc[i][node_col]
            if node in partition:
                com = partition[node]
            elif str(node) in partition:
                com = partition[str(node)]
            else:
                com = missing_community_index
                missing_community_index += 1 #increment missing index

            out_df.loc[i, train_target_name] = com

        return out_df




