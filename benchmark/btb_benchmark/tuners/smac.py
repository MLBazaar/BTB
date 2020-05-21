# -*- coding: utf-8 -*-
"""SMAC: Sequential Model-based Algorithm Configuration"""

import ConfigSpace.hyperparameters as hp
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario


def create_config_space(dict_hyperparams):
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
            hp_default = hyperparam.get('default') or hp_range[0]
            config_space.add_hyperparameter(
                hp.CategoricalHyperparameter(name, hp_range, default_value=hp_default))

    return config_space


def adapt_scoring_function(scoring_function):
    """Adapt the scoring function.

    SMAC optimize function calls the scoring function with a dict object, however our challenges
    recieve them as kwargs. The optimize function is ment to minimize, thats why we also return
    a negative score.
    """

    def adapted_function(params):
        return -scoring_function(**params)

    return adapted_function


def smac_tuning_function(optimizer, scoring_function, tunable_hyperparameters, iterations):
    """Construct and run a full optimization loop.

    Convert the hyperparameters in to the configuration space that smac expects.

    Args:
        scoring_function (function):
            A function that performs scoring over params.
        tunable_hyperparameters (dict):
            A python dict with hyperparameters.
        iterations (int):
            Number of tuning iterations to perform.
    """

    config_space = create_config_space(tunable_hyperparameters)
    tae_runner = adapt_scoring_function(scoring_function)
    scenario = Scenario({
        'run_obj': 'quality',
        'runcount-limit': iterations,
        'cs': config_space,
        'deterministic': 'true',
    })

    smac = optimizer(scenario=scenario, rng=42, tae_runner=tae_runner)
    best_config = smac.optimize()

    return scoring_function(**best_config)


def smac_smac4hpo_tuning_function(scoring_function, tunable_hyperparameters, iterations):
    return smac_tuning_function(SMAC4HPO, scoring_function, tunable_hyperparameters, iterations)
