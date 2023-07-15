# -*- coding: utf-8 -*-

from btb_benchmark.tuning_functions.ax import ax_optimize
from btb_benchmark.tuning_functions.btb import (
    gcpeituner, gcptuner, gpeituner, gptuner, uniformtuner)
from btb_benchmark.tuning_functions.hyperopt import hyperopt_tpe
from btb_benchmark.tuning_functions.skopt import skopt_EI, skopt_gp_hedge, skopt_LCB, skopt_PI
from btb_benchmark.tuning_functions.smac import (
    smac_hb4ac, smac_smac4hpo_ei, smac_smac4hpo_lcb, smac_smac4hpo_pi)


def get_all_tuning_functions():
    """Return all the tuning functions ready to use with benchmark."""
    return {
        "Ax.optimize": ax_optimize,
        "baytune.GPTuner": gptuner,
        "baytune.GCPTuner": gcptuner,
        "baytune.GPEiTuner": gpeituner,
        "baytune.GCPEiTuner": gcpeituner,
        "baytune.UniformTuner": uniformtuner,
        "HyperOpt.tpe": hyperopt_tpe,
        "skopt.EI": skopt_EI,
        "skopt.gp_hedge": skopt_gp_hedge,
        "skopt.LCB": skopt_LCB,
        "skopt.PI": skopt_PI,
        "SMAC.HB4AC": smac_hb4ac,
        "SMAC.SMAC4HPO_EI": smac_smac4hpo_ei,
        "SMAC.SMAC4HPO_LCB": smac_smac4hpo_lcb,
        "SMAC.SMAC4HPO_PI": smac_smac4hpo_pi,
    }
