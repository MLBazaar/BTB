# -*- coding: utf-8 -*-

from btb.benchmark.tuners.btb import make_btb_tuning_function
from btb.benchmark.tuners.hyperopt import hyperopt_tuning_function
from btb.tuning.tuners import GPEiTuner, GPTuner, UniformTuner


def get_all_tuning_functions():
    """Return all the tuning functions ready to use with benchmark."""
    return {
        'GPTuner': make_btb_tuning_function(GPTuner),
        'GPEiTuner': make_btb_tuning_function(GPEiTuner),
        'UniformTuner': make_btb_tuning_function(UniformTuner),
        'HyperOpt': hyperopt_tuning_function,
    }
