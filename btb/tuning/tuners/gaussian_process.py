
from btb.tuning.acquisition.numpyargmax import NumpyArgMaxFunction
from btb.tuning.metamodels.gaussian_process import GaussianProcessMetaModel, RandomForestMetaModel
from btb.tuning.tuners.base import BaseMetaModelTuner


class GaussianProcessTuner(GaussianProcessMetaModel, NumpyArgMaxFunction, BaseMetaModelTuner):
    pass


class GaussianProcessAlphaTuner(GaussianProcessMetaModel, NumpyArgMaxFunction, BaseMetaModelTuner):
    def __init__(self, tunable, alpha=0.1, num_candidates=1000):
        super().__init__(tunable, num_candidates)
        self._model_kwargs = {'alpha': alpha}


class RandomForestTuner(RandomForestMetaModel, NumpyArgMaxFunction, BaseMetaModelTuner):
    pass
