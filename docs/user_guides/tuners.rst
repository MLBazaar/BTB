Tuners
------

Tuners are specifically designed to speed up the process of selecting the
optimal hyper parameter values for a specific machine learning algorithm.

``btb.tuning.tuners`` defines Tuners: classes with a fit/predict/propose interface for
suggesting sets of hyperparameters.

This is done by following a Bayesian Optimization approach and iteratively:

* letting the tuner propose new sets of hyper parameter
* fitting and scoring the model with the proposed hyper parameters
* passing the score obtained back to the tuner

At each iteration the tuner will use the information already obtained to propose
the set of hyper parameters that it considers that have the highest probability
to obtain the best results.

To instantiate a ``Tuner`` all we need is a ``Tunable`` class with a collection of
``hyperparameters``.

.. ipython:: python

    from btb.tuning import Tunable
    from btb.tuning.tuners import GPTuner
    from btb.tuning.hyperparams import IntHyperParam

    hyperparams = {
        'n_estimators': IntHyperParam(min=10, max=500),
        'max_depth': IntHyperParam(min=10, max=500),
    }

    tunable = Tunable(hyperparams)
    tuner = GPTuner(tunable)

Then we perform the following three steps in a loop.

1. Let the Tuner propose a new set of parameters:

    .. ipython:: python

        parameters = tuner.propose()
        parameters

2. Fit and score a new model using these parameters:

    .. ipython:: python

        model = RandomForestClassifier(**parameters)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        score

3. Pass the used parameters and the score obtained back to the tuner:

    .. ipython:: python

        tuner.record(parameters, score)

At each iteration, the ``Tuner`` will use the information about the previous tests
to evaluate and propose the set of parameter values that have the highest probability
of obtaining the highest score.
