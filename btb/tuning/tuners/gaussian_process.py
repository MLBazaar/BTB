# -*- coding: utf-8 -*-

"""Package where the tuners based on GaussianProcessMetaModel are defined."""

from btb.tuning.acquisition.argsort import ArgSortAcquisition
from btb.tuning.acquisition.expected_improvement import ExpectedImprovementAcquisition
from btb.tuning.metamodels.gaussian_process import GaussianProcessMetaModel
from btb.tuning.tuners.base import BaseMetaModelTuner


class GPTuner(GaussianProcessMetaModel, ArgSortAcquisition, BaseMetaModelTuner):
    """GaussianProcess Tuner.

    This class uses a ``GaussianProcessRegressor`` model from the ``sklearn.gaussian_process``
    package, using a ``numpy.argmax`` function to return the better configurations predicted
    from the model.
    """
    pass


class GPEiTuner(GaussianProcessMetaModel, ExpectedImprovementAcquisition, BaseMetaModelTuner):
    """GaussianProcess Tuner.

    This class uses a ``GaussianProcessRegressor`` model from the ``sklearn.gaussian_process``
    package, using an ``ExpectedImprovement`` function to return the better configurations
    predicted from the model.
    """
    pass
