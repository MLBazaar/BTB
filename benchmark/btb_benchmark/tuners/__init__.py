# -*- coding: utf-8 -*-

from btb_benchmark.tuners.ax import ax_tuning_function
from btb_benchmark.tuners.btb import (
    gpeituner_tuning_function, gptuner_tuning_function, uniformtuner_tuning_function)
from btb_benchmark.tuners.hyperopt import (
    hyperopt_rand_tuning_function, hyperopt_tpe_tuning_function)


def get_all_tuners():
    """Return all the tuning functions ready to use with benchmark."""
    return {
        'Ax.optimize': ax_tuning_function,
        'BTB.GPTuner': gptuner_tuning_function,
        'BTB.GPEiTuner': gpeituner_tuning_function,
        'BTB.UniformTuner': uniformtuner_tuning_function,
        'HyperOpt.tpe': hyperopt_tpe_tuning_function,
        'HyperOpt.rand': hyperopt_rand_tuning_function,
    }
