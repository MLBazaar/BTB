from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from btb import HyperParameter, ParamTypes
from btb.tuning import GP
from btb.selection import Selector

"""
Selection example of chosing whether to give tuning budget to a Random Foreset
pipeline or an SVM pipeline.
We use a Selector to decide whether to tune the RF or SVM pipeline next.
We use a GP-based tuner for both pipelines.

We tune the n_estimators and max_depth parameters of the Random Forrest.
We tune tge c and gamma parameters of the SVM.
"""

def tune_pipeline(
    X,
    y,
    X_val,
    y_val,
    generate_model,
    tuner,
):
    """
    Tunes a specified pipeline with the
    specified tuner for TUNING_BUDGET_PER_ITER (3) iterations.

    Params:
        X: np.array of X training data
        y: np.array of y training data
        X_val: np.array of X validation data
        y_val: np.array of y validation data
        generate_model: function that returns an slkearn model to fit
        tuner: BTB tuner object for tuning hyperparameters
    """
    best_so_far = max(tuner.y) if len(tuner.y) >0 else 0 # keep track of best score
    print("Tuning with GP tuner for %s iterations"%TUNING_BUDGET_PER_ITER)
    for i in range(TUNING_BUDGET_PER_ITER):
        params =  tuner.propose()
        # create model using proposed hyperparams from tuner
        model = generate_model(params)
        model.fit(X, y)
        predicted = model.predict(X_val)
        score = accuracy_score(predicted, y_val)
        if score > best_so_far:
            best_so_far = score
            print("Improved pipeline to:", score)
        # record hyper-param combination and score for tuning
        tuner.add(params, score)
    print("Final score:", best_so_far)
    return best_so_far

if __name__ == '__main__':

    # Load data
    print("Loading MNIST Data........")
    mnist = fetch_mldata('MNIST original')
    X, X_test, y, y_test = train_test_split(
        mnist.data,
        mnist.target,
        train_size = 1000,
        test_size = 300,
    )

    # Establish global variables
    SELCTOR_NUM_ITER = 5 #we will use the selector 5 times
    TUNING_BUDGET_PER_ITER = 3 #we will tune for 3 iterations per round of selection

    # initialize the tuners
    # parameters of RandomForestClassifier we wish to tune and their ranges
    tunables_rf = [
        ('n_estimators', HyperParameter(ParamTypes.INT, [10, 500])),
        ('max_depth', HyperParameter(ParamTypes.INT, [3,20]))
    ]
    # parameters of SVM we wish to tune and their ranges
    tunables_svm = [
        ('c', HyperParameter(ParamTypes.FLOAT_EXP, [0.01, 10.0])),
        ('gamma', HyperParameter(ParamTypes.FLOAT, [0.000000001,0.0000001]))
    ]
    # Create a GP-based tuner for these tunables
    rf_tuner = GP(tunables_rf)
    svm_tuner = GP(tunables_svm)

    # Function to generate proper model given hyperparameters
    gen_rf = lambda params: RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            n_jobs=-1,
            verbose=False,
        )

    gen_svm = lambda params: SVC(
            C=params['c'],
            gamma=params['gamma'],
            max_iter=-1,
            verbose=False,
        )

    # Start by tuning each choice a few times, to generate scores and kick start
    # the selection
    print("---------Inital tuning of RF pipeline ---------")
    tune_pipeline(
        X,
        y,
        X_test,
        y_test,
        gen_rf,
        rf_tuner,
    )

    print("---------Inital tuning of SVM pipeline ---------")
    tune_pipeline(
        X,
        y,
        X_test,
        y_test,
        gen_svm,
        svm_tuner,
    )

    # Create a dictionary mapping each pipeline choice (RF, SVM) to its scores
    choice_scores = {'RF': rf_tuner.y, 'SVM': svm_tuner.y}

    # Create a selector for these two pipeline options
    selector = Selector(choice_scores.keys())

    for i in range(SELCTOR_NUM_ITER):
        # Using available score data, use Selector to choose next pipeline
        # to give tuning budge to
        next_pipeline = selector.select(choice_scores)
        print("\n---------SELECTED %s pipeline for tuning budget---------"%next_pipeline)
        if next_pipeline == 'RF':
            #give tuning budget Random Forrest pipeline
            tune_pipeline(
                X,
                y,
                X_test,
                y_test,
                gen_rf,
                rf_tuner,
            )
            choice_scores['RF'] = rf_tuner.y
        elif next_pipeline=='SVM':
            #give tuning budget to SVM pipeline
            tune_pipeline(
                X,
                y,
                X_test,
                y_test,
                gen_svm,
                svm_tuner,
            )
            choice_scores['SVM'] = svm_tuner.y

    # Out of tuning budget, report final scores for each pipeline
    print("---------------DONE---------------")
    print("Final score RF:", max(rf_tuner.y))
    print("Final score SVM:", max(svm_tuner.y))
