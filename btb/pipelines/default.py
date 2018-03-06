import pandas as pd
from .dm_pipeline import DmPipeline


class DefaultPipeline(DmPipeline):
    D3M_PRIMITIVES = []

    def fit(self, d3mds):
        "Fits model to given dataset for given hyperparameters"""
        task_type = d3mds.problem.get_taskType()
        task_sub_type = d3mds.problem.get_taskSubType()

        self.train_target_name = d3mds.problem.get_targets()[0]["colName"]
        train_targets = pd.Series(d3mds.get_train_targets()[:,0])  # get first target

        # supported_task_sub_types = ["binary", "multiClass", "uniVariate"]
        non_supported_task_sub_types = ["multiLabel",  "multiVariate", "overlapping", "nonOverlapping"]
        if task_sub_type in non_supported_task_sub_types:
            print("Unsupported task sub type")
            use_default_prediction = True

        # supported_task_types = ["classification", "regression"]
        non_supported_task_types = ["timeseriesForecasting", "similarityMatching", "linkPrediction", "vertexNomination", "communityDetection", "graphMatching", "collaborativeFiltering"]
        default_prediction = 0
        if task_type == "classification":
            default_prediction = train_targets.mode().iloc[0]
        elif task_type == "regression":
            default_prediction = train_targets.median()
        elif task_type in non_supported_task_types:
            print("Unsupported task type")
            use_default_prediction = True
            if train_targets is not None:
                if task_type == "linkPrediction":
                    default_prediction = 0
                elif task_type == "collaborativeFiltering":
                    default_prediction = train_targets.median()
                elif task_type == "graphMatching":
                    default_prediction = train_targets.mode().iloc[0]
                    try:
                        default_prediction = train_targets.median()
                    except:
                        try:
                            default_prediction = train_targets.mode().iloc[0]
                        except:
                            pass
        else:
            print("Unrecognized task type")


        print((("Default prediction: %s") % str(default_prediction)))
        self.default_prediction = default_prediction

    def predict(self, d3mds):
        """Make predictions on fitted pipeline"""
        fm = d3mds.get_data_all(dropTargets=True)
        print("Making a constant prediction")
        out_df = pd.DataFrame()
        out_df["d3mIndex"] = fm.index.values
        out_df[self.train_target_name] = self.default_prediction
        return out_df
