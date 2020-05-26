# -*- coding: utf-8 -*-

from hyperopt import rand, tpe

from btb.tuning.tuners import GPEiTuner, GPTuner, UniformTuner
from btb_benchmark.tuners.ax import ax_tuning_function
from btb_benchmark.tuners.btb import make_btb_tuning_function
from btb_benchmark.tuners.hyperopt import make_hyperopt_tuning_function
from btb_benchmark.tuners.smac import (smac_hb4ac, smac_smac4hpo_ei,
                                       smac_smac4hpo_lcb, smac_smac4hpo_pi)


def get_all_tuners():
    """Return all the tuning functions ready to use with benchmark."""
    return {
        'Ax.optimize': ax_tuning_function,
        'BTB.GPTuner': make_btb_tuning_function(GPTuner),
        'BTB.GPEiTuner': make_btb_tuning_function(GPEiTuner),
        'BTB.UniformTuner': make_btb_tuning_function(UniformTuner),
        'HyperOpt.tpe': make_hyperopt_tuning_function(tpe.suggest),
        'HyperOpt.rand': make_hyperopt_tuning_function(rand.suggest),
        'SMAC.HB4AC': smac_hb4ac,
        'SMAC.SMAC4HPO_EI': smac_smac4hpo_ei,
        'SMAC.SMAC4HPO_LCB': smac_smac4hpo_lcb,
        'SMAC.SMAC4HPO_PI': smac_smac4hpo_pi,
    }
