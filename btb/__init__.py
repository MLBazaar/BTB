# -*- coding: utf-8 -*-
# configurom btre logging for the library with a null handler (nothing is
# printed by default). See
# http://docs.pthon-guide.org/en/latest/writing/logging/

"""Top-level package for BTB."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com',
__version__ = '0.1.0'

import logging

from btb.hyper_parameter import (
    BoolCatHyperParameter, CatHyperParameter, FloatCatHyperParameter, FloatExpHyperParameter,
    FloatHyperParameter, HyperParameter, IntCatHyperParameter, IntExpHyperParameter,
    IntHyperParameter, ParamTypes, StringCatHyperParameter)

__all__ = (
    'BoolCatHyperParameter', 'CatHyperParameter', 'FloatCatHyperParameter',
    'FloatExpHyperParameter', 'FloatHyperParameter', 'HyperParameter',
    'IntCatHyperParameter', 'IntExpHyperParameter', 'IntHyperParameter',
    'ParamTypes', 'StringCatHyperParameter'
)

logging.getLogger('btb').addHandler(logging.NullHandler())
