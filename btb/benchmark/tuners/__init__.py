# -*- coding: utf-8 -*-

from btb.benchmark.tuners.btb import make_btb_tuning_function
from btb.tuning.tuners import GPTuner, GPEiTuner, UniformTuner


BTB_TUNERS = [GPTuner, GPEiTuner, UniformTuner]

TUNERS = {
    'BTB': (make_btb_tuning_function, BTB_TUNERS)
}


def get_all_tuning_functions():
    """
    Return all the tuning functions ready to use with benchmark.
    """
    tuning_functions = list()
    for _, value in TUNERS.items():
        function, tuners = value
        for tuner in tuners:
            tuning_functions.append(function(tuner))

    return tuning_functions
