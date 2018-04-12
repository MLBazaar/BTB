# configurom btre logging for the library with a null handler (nothing is
# printed by default). See
# http://docs.pthon-guide.org/en/latest/writing/logging/


__version__ = '0.2.0'

import logging

from btb.hyper_parameter import (
    CLASS_GENERATOR, BoolCatHyperParameter, CatHyperParameter, FloatCatHyperParameter,
    FloatExpHyperParameter, FloatHyperParameter, HyperParameter, IntCatHyperParameter,
    IntExpHyperParameter, IntHyperParameter, ParamTypes, StringCatHyperParameter)

__all__ = (
    'CLASS_GENERATOR', 'BoolCatHyperParameter', 'CatHyperParameter',
    'FloatCatHyperParameter', 'FloatExpHyperParameter',
    'FloatHyperParameter', 'HyperParameter', 'IntCatHyperParameter',
    'IntExpHyperParameter', 'IntHyperParameter', 'ParamTypes',
    'StringCatHyperParameter'
)

logging.getLogger('btb').addHandler(logging.NullHandler())
