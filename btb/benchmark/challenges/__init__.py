# -*- coding: utf-8 -*-

"""Top level where all the challenges are imported."""

from btb.benchmark.challenges.bohachevsky import Bohachevsky
from btb.benchmark.challenges.boston import BostonABR, BostonBR, BostonRFR
from btb.benchmark.challenges.branin import Branin
from btb.benchmark.challenges.census import CensusABC, CensusRFC, CensusSGDC
from btb.benchmark.challenges.rosenbrock import Rosenbrock
from btb.benchmark.challenges.wind import WindABC, WindRFC, WindSGDC

__all__ = (
    'Bohachevsky',
    'BostonABR',
    'BostonBR',
    'BostonRFR',
    'Branin',
    'CensusABC',
    'CensusRFC',
    'CensusSGDC',
    'Rosenbrock',
    'WindABC',
    'WindRFC',
    'WindSGDC',
)
