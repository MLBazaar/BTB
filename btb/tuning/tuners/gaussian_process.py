# -*- coding: utf-8 -*-

"""Package where the tuners based on GaussianProcessMetaModel are defined."""

from btb.tuning.acquisition.expected_improvement import ExpectedImprovementFunction
from btb.tuning.acquisition.numpyargmax import NumpyArgMaxFunction
from btb.tuning.metamodels.gaussian_process import GaussianProcessMetaModel
from btb.tuning.tuners.base import BaseMetaModelTuner


class GPTuner(GaussianProcessMetaModel, NumpyArgMaxFunction, BaseMetaModelTuner):
    pass


class GPEiTuner(GaussianProcessMetaModel, ExpectedImprovementFunction, BaseMetaModelTuner):
    pass
