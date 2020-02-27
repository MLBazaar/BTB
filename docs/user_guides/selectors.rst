Selectors
---------

The selectors are intended to be used in combination with tuners in order to find
out and decide which model seems to get the best results once it is properly fine tuned.

In order to use the selector we will create a ``Tuner`` instance for each model that
we want to try out, as well as the ``Selector`` instance.

.. ipython:: python

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import make_scorer, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.svm import SVC

    from btb.selection import UCB1
    from btb.tuning.hyperparams import FloatHyperParam, IntHyperParam
    from btb.tuning.tunable import Tunable
    from btb.tuning.tuners import GPTuner

    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.3, random_state=0)

    models = {
        'RF': RandomForestClassifier,
        'SVC': SVC
    }

    selector = UCB1(['RF', 'SVC'])
    rf_hyperparams = {
        'n_estimators': IntHyperParam(min=10, max=500),
        'max_depth': IntHyperParam(min=3, max=20)
    }
    rf_tunable = Tunable(rf_hyperparams)
    svc_hyperparams = {
        'C': FloatHyperParam(min=0.01, max=10.0),
        'gamma': FloatHyperParam(0.000000001, 0.0000001)
    }
    svc_tunable = Tunable(svc_hyperparams)
    tuners = {
        'RF': GPTuner(rf_tunable),
        'SVC': GPTuner(svc_tunable)
    }

Then we perform the following steps in a loop.

1. Pass all the obtained scores to the selector and let it decide which model to test.

    .. ipython:: python

        next_choice = selector.select({
            'RF': tuners['RF'].scores,
            'SVC': tuners['SVC'].scores
        })
        next_choice

2. Obtain a new set of parameters from the indicated tuner and create a model instance.

    .. ipython:: python

        parameters = tuners[next_choice].propose()
        model = models[next_choice](**parameters)

3. Evaluate the score of the new model instance and pass it back to the tuner

    .. ipython:: python

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        tuners[next_choice].record(parameters, score)
