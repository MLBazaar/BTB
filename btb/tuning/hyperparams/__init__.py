# -*- coding: utf-8 -*-

"""Top level where all the hyperparameters are imported."""

from btb.tuning.hyperparams.boolean import BooleanHyperParam
from btb.tuning.hyperparams.categorical import CategoricalHyperParam
from btb.tuning.hyperparams.numerical import FloatHyperParam, IntHyperParam

__all__ = ('BooleanHyperParam', 'CategoricalHyperParam', 'FloatHyperParam', 'IntHyperParam')


def instantiate_hyperparam_from_dict(dict_hyperparam):
    """Instantiate a hyperparameter from a dictionary"""

    if not isinstance(dict_hyperparam, dict):
        raise TypeError('Hyperparams must be a dictionary.')

    hp_type = dict_hyperparam['type']
    hp_default = dict_hyperparam.get('default')

    if hp_type == 'int':
        hp_range = dict_hyperparam.get('range') or dict_hyperparam.get('values')
        hp_min = min(hp_range) if hp_range else None
        hp_max = max(hp_range) if hp_range else None
        hp_instance = IntHyperParam(min=hp_min, max=hp_max, default=hp_default)

    elif hp_type == 'float':
        hp_range = dict_hyperparam.get('range') or dict_hyperparam.get('values')
        hp_min = min(hp_range)
        hp_max = max(hp_range)
        hp_instance = FloatHyperParam(min=hp_min, max=hp_max, default=hp_default)

    elif hp_type == 'bool':
        hp_instance = BooleanHyperParam(default=hp_default)

    elif hp_type == 'str':
        hp_choices = dict_hyperparam.get('range') or dict_hyperparam.get('values')
        hp_instance = CategoricalHyperParam(choices=hp_choices, default=hp_default)

    return hp_instance
