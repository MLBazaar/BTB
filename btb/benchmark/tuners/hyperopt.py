# -*- coding: utf-8 -*-

from hyperopt import Trials, fmin, hp


def search_space_from_dict(dict_hyperparams):
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


def make_minimize_function(scoring_function):
    """Convert scoring function to minimize the score.

    As ``BTB`` works with maximization, we have created all our challenges to ``maximize`` the
    score and ``HyperOpt`` works with minimization only.

    Also ``hyperopt`` params are being passed as an python ``dict`` and we adapt those to be
    passed as ``kwargs``.
    """
    def minimized_function(params):
        return -scoring_function(**params)

    return minimized_function


def make_hyperopt_tuning_function(algo):
    """Create a hyperopt minimize tuning function.

    Args:
        algo (hyperopt.algo):
            Search Hyperopt Algorithm to be used with ``fmin`` function.
    """
    def hyperopt_tuning_function(scoring_function, tunable_hyperparameters, iterations):

        minimized_scoring = make_minimize_function(scoring_function)
        search_space = search_space_from_dict(tunable_hyperparameters)
        trials = Trials()
        fmin(minimized_scoring, search_space, algo=algo, max_evals=iterations, trials=trials)

        # normalize best score to match other tuners
        best_score = -1 * trials.best_trial['result']['loss']

        return best_score

    return hyperopt_tuning_function
