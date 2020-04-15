# -*- coding: utf-8 -*-

from hyperopt import rand, tpe

from btb.tuning.tuners import GPEiTuner, GPTuner, UniformTuner
from btb_benchmark.tuners.btb import make_btb_tuning_function
from btb_benchmark.tuners.hyperopt import make_hyperopt_tuning_function


def get_all_tuners():
    """Return all the tuning functions ready to use with benchmark."""
    return {
        'BTB.GPTuner': make_btb_tuning_function(GPTuner),
        'BTB.GPEiTuner': make_btb_tuning_function(GPEiTuner),
        'BTB.UniformTuner': make_btb_tuning_function(UniformTuner),
        'HyperOpt.tpe': make_hyperopt_tuning_function(tpe.suggest),
        'HyperOpt.rand': make_hyperopt_tuning_function(rand.suggest),
    }
