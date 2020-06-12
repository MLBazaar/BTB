# -*- coding: utf-8 -*-

from ax import optimize


def convert_hyperparameters(dict_hyperparams):
    """Convert the hyperparameters to the format accepted by the library."""
    hyperparams = []

    if not isinstance(dict_hyperparams, dict):
        raise TypeError('Hyperparams must be a dictionary.')

    for name, hyperparam in dict_hyperparams.items():
        hp_type = hyperparam['type']

        if hp_type == 'int':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = int(min(hp_range))
            hp_max = int(max(hp_range))
            hyperparams.append({
                'name': name,
                'type': 'range',
                'bounds': [hp_min, hp_max]
            })

        elif hp_type == 'float':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = float(min(hp_range))
            hp_max = float(max(hp_range))
            hyperparams.append({
                'name': name,
                'type': 'range',
                'bounds': [hp_min, hp_max]
            })

        elif hp_type == 'bool':
            hyperparams.append({
                'name': name,
                'type': 'choice',
                'bounds': [True, False]
            })

        elif hp_type == 'str':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hyperparams.append({
                'name': name,
                'type': 'choice',
                'bounds': hp_range,
            })

    return hyperparams


def adapt_scoring_function(scoring_function):
    """Adapt the scoring function.

    Ax's optimize function calls the scoring function with a dict object, however our challenges
    recieve them as kwargs.
    """
    def adapted_function(params):
        return scoring_function(**params)

    return adapted_function


def ax_optimize(scoring_function, tunable_hyperparameters, iterations):
    """Construct and run a full optimization loop.

    Convert the hyperparameters in to the accepted format by ``Ax`` and adapt the scoring
    function to pass the configuration as kwargs.

    Args:
        scoring_function (function):
            A function that performs scoring over params.
        tunable_hyperparameters (dict):
            A python dict with hyperparameters.
        iterations (int):
            Number of tuning iterations to perform.
    """
    parameters = convert_hyperparameters(tunable_hyperparameters)
    evaluation_function = adapt_scoring_function(scoring_function)

    best_params = optimize(
        parameters=parameters,
        evaluation_function=evaluation_function,
        total_trials=iterations,
        minimize=False
    )[0]

    return evaluation_function(best_params)
