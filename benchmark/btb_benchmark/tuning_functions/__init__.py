# -*- coding: utf-8 -*-

from btb_benchmark.tuning_functions.ax import ax_optimize
from btb_benchmark.tuning_functions.btb import gpeituner, gptuner, uniformtuner
from btb_benchmark.tuning_functions.hyperopt import hyperopt_tpe
from btb_benchmark.tuning_functions.smac import (
    smac_hb4ac, smac_smac4hpo_ei, smac_smac4hpo_lcb, smac_smac4hpo_pi)


def get_all_tuning_functions():
    """Return all the tuning functions ready to use with benchmark."""
    return {
        'Ax.optimize': ax_optimize,
        'BTB.GPTuner': gptuner,
        'BTB.GPEiTuner': gpeituner,
        'BTB.UniformTuner': uniformtuner,
        'HyperOpt.tpe': hyperopt_tpe,
        'SMAC.HB4AC': smac_hb4ac,
        'SMAC.SMAC4HPO_EI': smac_smac4hpo_ei,
        'SMAC.SMAC4HPO_LCB': smac_smac4hpo_lcb,
        'SMAC.SMAC4HPO_PI': smac_smac4hpo_pi,
    }
