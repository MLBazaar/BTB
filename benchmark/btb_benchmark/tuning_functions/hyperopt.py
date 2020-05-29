# -*- coding: utf-8 -*-

from hyperopt import Trials, fmin, hp, tpe


def _search_space_from_dict(dict_hyperparams):
    hyperparams = {}

    if not isinstance(dict_hyperparams, dict):
        raise TypeError('Hyperparams must be a dictionary.')

    for name, hyperparam in dict_hyperparams.items():
        hp_type = hyperparam['type']

        if hp_type == 'int':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = min(hp_range) if hp_range else None
            hp_max = max(hp_range) if hp_range else None
            hp_instance = hp.uniformint(name, hp_min, hp_max)

        elif hp_type == 'float':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = min(hp_range)
            hp_max = max(hp_range)
            hp_instance = hp.uniform(name, hp_min, hp_max)

        elif hp_type == 'bool':
            hp_instance = hp.choice(name, [True, False])

        elif hp_type == 'str':
            hp_choices = hyperparam.get('range') or hyperparam.get('values')
            hp_instance = hp.choice(name, hp_choices)

        hyperparams[name] = hp_instance

    return hyperparams


def _make_minimize_function(scoring_function):
    """Create a minimize function.

    Given a maximization ``scoring_function`` convert it to minimize in order to work with
    ``hyperopt``, as ``benchmark`` works with ``maximization``.

    Also ``hyperopt`` params are being passed as a python ``dict``, we pass those as ``kwargs``
    to the ``scoring_function``.
    """
    def minimized_function(params):
        return -scoring_function(**params)

    return minimized_function


def _hyperopt_tuning_function(algo, scoring_function, tunable_hyperparameters, iterations):
    """Create a tuning function that uses ``HyperOpt``.

    With a given suggesting algorithm from the library ``HyperOpt``, create a tuning
    function that maximize the score, using ``fmin``.

    Args:
        algo (hyperopt.algo):
            Search / Suggest ``HyperOpt`` algorithm to be used with ``fmin`` function.
    """

    minimized_scoring = _make_minimize_function(scoring_function)
    search_space = _search_space_from_dict(tunable_hyperparameters)
    trials = Trials()
    fmin(
        minimized_scoring,
        search_space,
        algo=algo,
        max_evals=iterations,
        trials=trials,
        verbose=False
    )

    # normalize best score to match other tuners
    best_score = -1 * trials.best_trial['result']['loss']

    return best_score


def hyperopt_tpe(scoring_function, tunable_hyperparameters, iterations):
    """Tree-structured Parzen Estimator"""
    return _hyperopt_tuning_function(
        tpe.suggest,
        scoring_function,
        tunable_hyperparameters,
        iterations
    )
