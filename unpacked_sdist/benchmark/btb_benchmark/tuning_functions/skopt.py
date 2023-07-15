# -*- coding: utf-8 -*-

from skopt import gp_minimize, space
from skopt.utils import use_named_args


def _dimension_space_from_dict(dict_hyperparams):
    hyperparams = list()

    if not isinstance(dict_hyperparams, dict):
        raise TypeError('Hyperparams must be a dictionary.')

    for name, hyperparam in dict_hyperparams.items():
        hp_type = hyperparam['type']

        if hp_type == 'int':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = min(hp_range) if hp_range else None
            hp_max = max(hp_range) if hp_range else None
            hp_instance = space.Integer(hp_min, hp_max, name=name)

        elif hp_type == 'float':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = min(hp_range)
            hp_max = max(hp_range)
            hp_instance = space.Real(hp_min, hp_max, name=name)

        elif hp_type == 'bool':
            hp_instance = space.Categorical([True, False], name=name)

        elif hp_type == 'str':
            hp_choices = hyperparam.get('range') or hyperparam.get('values')
            hp_instance = space.Categorical(hp_choices, name=name)

        hyperparams.append(hp_instance)

    return hyperparams


def _make_minimize_function(scoring_function, space):
    """Create a minimize function.

    Given a maximization ``scoring_function`` convert it to minimize in order to work with
    ``hyperopt``, as ``benchmark`` works with ``maximization``.

    Also ``hyperopt`` params are being passed as a python ``dict``, we pass those as ``kwargs``
    to the ``scoring_function``.
    """
    @use_named_args(space)
    def minimized_function(**params):
        return -scoring_function(**params)

    return minimized_function


def _skopt_tuning_function(scoring_function, tunable_hyperparameters, iterations, acq_function):

    space = _dimension_space_from_dict(tunable_hyperparameters)
    scoring_function = _make_minimize_function(scoring_function, space)

    res = gp_minimize(
        scoring_function,
        space,
        acq_func=acq_function,
        n_calls=iterations,
        random_state=0
    )
    return -res.fun


def skopt_LCB(scoring_function, tunable_hyperparameters, iterations):
    """Lower Confidence Bound acquisition function."""
    return _skopt_tuning_function(
        scoring_function,
        tunable_hyperparameters,
        iterations,
        'LCB',
    )


def skopt_EI(scoring_function, tunable_hyperparameters, iterations):
    """Expected Improvement acquisition function."""
    return _skopt_tuning_function(
        scoring_function,
        tunable_hyperparameters,
        iterations,
        'EI',
    )


def skopt_PI(scoring_function, tunable_hyperparameters, iterations):
    """Probability of Improvement acquisition function."""
    return _skopt_tuning_function(
        scoring_function,
        tunable_hyperparameters,
        iterations,
        'PI',
    )


def skopt_gp_hedge(scoring_function, tunable_hyperparameters, iterations):
    """Probabilistically choose between `LCB`, `EI` or `PI` acquisition function while tuning."""
    return _skopt_tuning_function(
        scoring_function,
        tunable_hyperparameters,
        iterations,
        'gp_hedge',
    )
