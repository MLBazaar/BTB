# -*- coding: utf-8 -*-
"""SMAC: Sequential Model-based Algorithm Configuration"""

from tempfile import TemporaryDirectory

import ConfigSpace.hyperparameters as hp
from smac.configspace import ConfigurationSpace
from smac.facade.hyperband_facade import HB4AC
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.optimizer import acquisition
from smac.scenario.scenario import Scenario

_NONE = '__NONE__'


def _create_config_space(dict_hyperparams):
    """Create the hyperparameters hyperspace."""
    config_space = ConfigurationSpace()

    if not isinstance(dict_hyperparams, dict):
        raise TypeError('Hyperparams must be a dictionary.')

    for name, hyperparam in dict_hyperparams.items():
        hp_type = hyperparam['type']

        if hp_type == 'int':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = min(hp_range)
            hp_max = max(hp_range)
            hp_default = hyperparam.get('default') or hp_min
            config_space.add_hyperparameter(
                hp.UniformIntegerHyperparameter(name, hp_min, hp_max, default_value=hp_default))

        elif hp_type == 'float':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = min(hp_range)
            hp_max = max(hp_range)
            hp_default = hyperparam.get('default') or hp_min
            config_space.add_hyperparameter(
                hp.UniformFloatHyperparameter(name, hp_min, hp_max, default_value=hp_default))

        elif hp_type == 'bool':
            hp_default = bool(hyperparam.get('default'))
            config_space.add_hyperparameter(
                hp.CategoricalHyperparameter(name, ['true', 'false'], default_value=hp_default))

        elif hp_type == 'str':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_range = [_NONE if hp is None else hp for hp in hp_range]
            hp_default = hyperparam.get('default') or hp_range[0]
            hp_default = _NONE if hp_default is None else hp_default

            config_space.add_hyperparameter(
                hp.CategoricalHyperparameter(name, hp_range, default_value=hp_default))

    return config_space


def _parse_params(params):
    parsed_params = dict()
    params = params if isinstance(params, dict) else params.get_dictionary()

    for key, value in params.items():
        parsed_params[key] = None if value == _NONE else value

    return parsed_params


def _adapt_scoring_function(scoring_function):
    """Adapt the scoring function.

    SMAC optimize function calls the scoring function with a dict object, however our challenges
    recieve them as kwargs. The optimize function is ment to minimize, thats why we also return
    a negative score.
    """

    def adapted_function(params):
        parsed_params = _parse_params(params)
        return -scoring_function(**parsed_params)

    return adapted_function


def _get_optimizer_params(scoring_function, tunable_hyperparameters,
                          iterations, tmp_dir, **kwargs):
    config_space = _create_config_space(tunable_hyperparameters)
    tae_runner = _adapt_scoring_function(scoring_function)
    scenario = Scenario({
        'run_obj': 'quality',
        'runcount_limit': iterations,
        'cs': config_space,
        'deterministic': 'true',
        'output_dir': tmp_dir,
        'limit_resources': False,
    })

    optimizer_params = {
        'scenario': scenario,
        'rng': 42,
        'tae_runner': tae_runner,
    }

    if kwargs:
        optimizer_params.update(kwargs)

    return optimizer_params


def _smac_tuning_function(optimizer, scoring_function,
                          tunable_hyperparameters, iterations, **kwargs):
    """Construct and run a full optimization loop.

    Given an optimizer from ``smac.facade``, use it to perform a complete optimization for
    a given ``scoring_function``. This is achieved by creating a ``config_space`` from the
    tunable hyperparameters, adapting the scoring function to work with minimization, and
    then, create an instace scenario that with the config space and the amount of iterations.

    Finally we use the optimizer with the previously created sceneario and adapted scoring
    function to optimize this and obtain the best configuration for the given iterations.

    Args:
        optimizer (type):
            A ``smac.facade`` class that represents a tuner.
        scoring_function (function):
            A function that performs scoring over params.
        tunable_hyperparameters (dict):
            A python dict with hyperparameters.
        iterations (int):
            Number of tuning iterations to perform.
        kwargs (kwargs):
            Any additional configuration used by the optimizer can be passed as
            keyword args.
    """
    with TemporaryDirectory() as tmp_dir:
        optimizer_params = _get_optimizer_params(
            scoring_function,
            tunable_hyperparameters,
            iterations,
            tmp_dir,
            **kwargs
        )

        smac = optimizer(**optimizer_params)
        best_config = smac.optimize()

    return scoring_function(**_parse_params(best_config))


def smac_smac4hpo_ei(scoring_function, tunable_hyperparameters, iterations):
    """Use SMAC4HPO with Expected Improvement acquisition function."""
    return _smac_tuning_function(
        SMAC4HPO,
        scoring_function,
        tunable_hyperparameters,
        iterations,
        acquisition_function=acquisition.EI
    )


def smac_smac4hpo_lcb(scoring_function, tunable_hyperparameters, iterations):
    """Use SMAC4HPO with Lower Confidence Bound acquisition function."""
    return _smac_tuning_function(
        SMAC4HPO,
        scoring_function,
        tunable_hyperparameters,
        iterations,
        acquisition_function=acquisition.LCB
    )


def smac_smac4hpo_pi(scoring_function, tunable_hyperparameters, iterations):
    """Use SMAC4HPO with Probability of Improvement acquisition function."""
    return _smac_tuning_function(
        SMAC4HPO,
        scoring_function,
        tunable_hyperparameters,
        iterations,
        acquisition_function=acquisition.PI
    )


def smac_hb4ac(scoring_function, tunable_hyperparameters, iterations):
    """Use HyperBand implementation from SMAC."""
    intensifier_kwargs = {
        'initial_budget': 1,
        'max_budget': iterations
    }

    return _smac_tuning_function(
        HB4AC,
        scoring_function,
        tunable_hyperparameters,
        iterations,
        intensifier_kwargs=intensifier_kwargs
    )
