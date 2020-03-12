# -*- coding: utf-8 -*-

from btb.benchmark.tuners.btb import make_btb_tuning_function
from btb.tuning.tuners import GPEiTuner, GPTuner, UniformTuner

BTB_TUNERS = [GPTuner, GPEiTuner, UniformTuner]

TUNERS = {
    'BTB': (make_btb_tuning_function, BTB_TUNERS)
}


def get_all_tuning_functions():
    """
    Return all the tuning functions ready to use with benchmark.
    """
    tuning_functions = {}
    for _, value in TUNERS.items():
        function, tuner_classes = value
        for tuner_class in tuner_classes:
            tuning_functions[tuner_class.__name__] = function(tuner_class)

    return tuning_functions
