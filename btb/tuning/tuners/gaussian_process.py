# -*- coding: utf-8 -*-

"""Package where the tuners based on GaussianProcessMetaModel are defined."""

from btb.tuning.acquisition.expected_improvement import ExpectedImprovementAcquisition
from btb.tuning.acquisition.predicted_score import PredictedScoreAcquisition
from btb.tuning.metamodels.gaussian_process import GaussianProcessMetaModel
from btb.tuning.tuners.base import BaseMetaModelTuner


class GPTuner(GaussianProcessMetaModel, PredictedScoreAcquisition, BaseMetaModelTuner):
    """GaussianProcess Tuner.

    This class uses a ``GaussianProcessRegressor`` model from the ``sklearn.gaussian_process``
    package, using a ``numpy.argmax`` function to return the better configurations predicted
    from the model.
    """
    def __init__(self, tunable, length_scale=1, **kwargs):
        self._metamodel_kwargs = {'length_scale': length_scale}
        super().__init__(tunable, **kwargs)


class GPEiTuner(GaussianProcessMetaModel, ExpectedImprovementAcquisition, BaseMetaModelTuner):
    """GaussianProcess Tuner.

    This class uses a ``GaussianProcessRegressor`` model from the ``sklearn.gaussian_process``
    package, using an ``ExpectedImprovement`` function to return the better configurations
    predicted from the model.
    """
    def __init__(self, tunable, length_scale=1, **kwargs):
        self._metamodel_kwargs = {'length_scale': length_scale}
        super().__init__(tunable, **kwargs)
