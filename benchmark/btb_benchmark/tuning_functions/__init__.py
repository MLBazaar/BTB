# -*- coding: utf-8 -*-

from btb_benchmark.tuning_functions.ax import ax_optimize
from btb_benchmark.tuning_functions.btb import gpeituner, gptuner, uniformtuner
from btb_benchmark.tuning_functions.hyperopt import hyperopt_tpe


def get_all_tuning_functions():
    """Return all the tuning functions ready to use with benchmark."""
    return {
        'Ax.optimize': ax_optimize,
        'BTB.GPTuner': gptuner,
        'BTB.GPEiTuner': gpeituner,
        'BTB.UniformTuner': uniformtuner,
        'HyperOpt.tpe': hyperopt_tpe,
    }
