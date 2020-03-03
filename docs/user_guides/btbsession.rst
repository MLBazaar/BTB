BTBSession
==========

A ``BTBSession`` represents the process of selecting and tuning several tunables
until the best possible configuration for a specific ``scorer`` is found.

For this, a loop is run in which for each iteration a combination of a ``Selector`` and
``Tuner`` is used to decide which tunable to score next and with which hyperparameters.

While running, the ``BTBSession`` handles the errors discarding, if configured to do so,
the tunables that have reached as many errors as the user specified.

Below there is a short example using ``BTBSession`` to perform tuning over
``ExtraTreesRegressor`` and ``RandomForestRegressor`` ensemblers from `scikit-learn`_
and both of them are evaluated against the `Boston dataset`_ regression problem.

Let's import all the needed packages in order to run our code. We will import the ``load_boston``
from ``sklearn.datasets`` and two estimators (``ExtraTreesRegressor`` and ``RandomForest``). In
order to evaluate them we will use the ``r2_score``. We will also be importing some basics in order
to perform cross validation and split our data from the ``model_selection``. And finally we will
import our ``BTBSession``.

.. ipython:: python

    from sklearn.datasets import load_boston as load_dataset
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.metrics import make_scorer, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split

    from btb.session import BTBSession

Then we can create a dictionary to reffer to each model of our choice:

.. ipython:: python

    models = {
        'random_forest': RandomForestRegressor,
        'extra_trees': ExtraTreesRegressor,
    }

Then we will proceed to create a function that will *build the model* and another one that will
*score* the model. This is needed because ``BTBSession.score`` is being called with the
``tunable_name`` and ``config``. So pretty much the scoring function must be prepared to recive
this two parameters and generate a ``score``.

.. ipython:: python

    def build_model(name, hyperparameters):
        model_class = models[name]
        return model_class(random_state=0, **hyperparameters)

    def score_model(name, hyperparameters):
        model = build_model(name, hyperparameters)
        r2_scorer = make_scorer(r2_score)
        scores = cross_val_score(model, X_train, y_train, scoring=r2_scorer, cv=5)
        return scores.mean()

Once we have this two functions, we can proceed on loading the dataset, generating its splits
and create a list of ``tunables`` for our ``BTBSession``.

.. ipython:: python

    dataset = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.3, random_state=0)

    tunables = {
        'random_forest': {
            'n_estimators': {
                'type': 'int',
                'default': 2,
                'range': [1, 1000]
            },
            'max_features': {
                'type': 'str',
                'default': 'log2',
                'range': [None, 'auto', 'log2', 'sqrt']
            },
            'min_samples_split': {
                'type': 'int',
                'default': 2,
                'range': [2, 20]
            },
            'min_samples_leaf': {
                'type': 'int',
                'default': 2,
                'range': [1, 20]
            },
        },
        'extra_trees': {
            'n_estimators': {
                'type': 'int',
                'default': 2,
                'range': [1, 1000]
            },
            'max_features': {
                'type': 'str',
                'default': 'log2',
                'range': [None, 'auto', 'log2', 'sqrt']
            },
            'min_samples_split': {
                'type': 'int',
                'default': 2,
                'range': [2, 20]
            },
            'min_samples_leaf': {
                'type': 'int',
                'default': 2,
                'range': [1, 20]
            },
        }
    }

Now that we have everything set up, we can proceed to generate our ``BTBSession`` and run it in
order to evaluate wich of this two machine learning models will obtain a better score.

.. ipython:: python

    session = BTBSession(tunables, score_model)
    session.run(5)

Once this 10 iterations are done, our session will return the ``best_proposal``, or we can access
it thro ``session.best_proposal``. Inside this dictionary we will find the ``name``, ``config`` and
the ``score`` for the best configuration found during those 10 iterations.

.. _you have already installed them: install.html#additional-dependencies
.. _scikit-learn: https://scikit-learn.org/
.. _Boston Dataset: http://lib.stat.cmu.edu/datasets/boston
