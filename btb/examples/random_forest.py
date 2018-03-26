from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from btb import HyperParameter, ParamTypes
from btb.tuning import GP, Uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

"""
Tuning example on a Random Forreset pipeline.
We compare the results of using a Uniform and GP-based tuner to tune a
sklearn RandomForestClassifier on the MNIST dataset.

We tune the n_estimators and max_depth parameters.
"""


def tune_random_forest(tuner, X, y, X_test, y_test):
    for i in range(10):
        params = tuner.propose()
        # create Random Forrest using proposed hyperparams from tuner
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            n_jobs=-1,
            verbose=False,
        )
        model.fit(X, y)
        predicted = model.predict(X_test)
        score = accuracy_score(predicted, y_test)
        # record hyper-param combination and score for tuning
        tuner.add(params, score)
    print("Final score:", tuner._best_score)


print("Loading MNIST Data")
mnist = fetch_mldata('MNIST original')
X, X_test, y, y_test = train_test_split(
    mnist.data,
    mnist.target,
    train_size=1000,
    test_size=300,
)

# parameters of RandomForestClassifier we wish to tune and their ranges
tunables = [
    ('n_estimators', HyperParameter(ParamTypes.INT, [10, 500])),
    ('max_depth', HyperParameter(ParamTypes.INT, [3, 20]))
]

print("-------Tuning with a Uniform Tuner-------")
tuner = Uniform(tunables)
tune_random_forest(tuner, X, y, X_test, y_test)

print("-------Tuning with a GP Tuner-------")
tuner = GP(tunables)
tune_random_forest(tuner, X, y, X_test, y_test)
