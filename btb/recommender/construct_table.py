import pandas as pd
import numpy as np
import math

FILE_PATH = 'classifiers.csv'

def construct_recommender_matrix():
    df = pd.read_csv(FILE_PATH)
    #todo reformat hyperparamtere values
    df['rounded_hyperparms'] = df.apply(reformat_hyperparameter_values, axis=1)
    df["classifierID"] = df.groupby(["rounded_hyperparms"]).grouper.group_info[0]
    num_unqiue_classifers = df['classifierID'].max()
    print("shape", df.shape)
    print("num unique", num_unqiue_classifers)
    df['classifier_score'] = list(zip(df.classifierID, df.rounded_hyperparms, df.test_judgment_metric))
    agg_datarun = df.groupby('datarun_id')['classifier_score'].apply(list)
    classifier_params = {}
    recommender_matrix = np.empty([len(agg_datarun), num_unqiue_classifers + 1])
    avg = 0
    for i in range(len(agg_datarun)):
        row = agg_datarun[i+1]
        avg += len(row)
        for classifier_data in row:
            classifier_id, hyperparams, score = classifier_data
            if classifier_id < 0:
                continue
            classifier_params[classifier_id] = hyperparams
            recommender_matrix[i][classifier_id] = score
    print("-----------recommender matrix-----------------")
    print(recommender_matrix)
    print(recommender_matrix.shape)
    print("---------classifier params-----------------")
    print(len(classifier_params.keys()))
    print("avg num classifiers", avg/len(agg_datarun))
    return recommender_matrix, classifier_params


def reformat_hyperparameter_values(row):
    unformated = row['hyperparameter_values']
    formatted = ''
    if unformated != unformated:
        return unformated
    key_values = unformated.split(';')[:-1]
    d = {}
    for each in key_values:
        t = each.split(":", 1)
        key = t[0]
        value = t[1]
        try:
            value = float(value)
            value = '%s' % float('%.1g' % value)
            formatted += key + ':' + value + ';'
        except:
            formatted += each + ';'
        d[key] = value
    formatted = ''
    for key in sorted(d):
        formatted += key + ':' + d[key] + ';'
    return formatted



if __name__ == '__main__':
    construct_recommender_matrix()
