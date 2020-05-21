# -*- coding: utf-8 -*-

from hyperopt import rand, tpe

from btb.tuning.tuners import GPEiTuner, GPTuner, UniformTuner
from btb_benchmark.tuners.ax import ax_tuning_function
from btb_benchmark.tuners.btb import make_btb_tuning_function
from btb_benchmark.tuners.hyperopt import make_hyperopt_tuning_function
from btb_benchmark.tuners.smac import smac_smac4bo_tuning_function, smac_smac4hpo_tuning_function


def get_all_tuners():
    """Return all the tuning functions ready to use with benchmark."""
    return {
        'Ax.optimize': ax_tuning_function,
        'BTB.GPTuner': make_btb_tuning_function(GPTuner),
        'BTB.GPEiTuner': make_btb_tuning_function(GPEiTuner),
        'BTB.UniformTuner': make_btb_tuning_function(UniformTuner),
        'HyperOpt.tpe': make_hyperopt_tuning_function(tpe.suggest),
        'HyperOpt.rand': make_hyperopt_tuning_function(rand.suggest),
        'SMAC.SMAC4BO': smac_smac4bo_tuning_function,
        'SMAC.SMAC4HPO': smac_smac4hpo_tuning_function,
    }
