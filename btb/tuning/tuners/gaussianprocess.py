
from btb.tuning.acquisition.numpyargmax import NumpyArgMaxFunction
from btb.tuning.metamodels.gausianprocess import GaussianProcessMetaModel
from btb.tuning.tuners.base import BaseMetaModelTuner


class GaussianProcess(GaussianProcessMetaModel, NumpyArgMaxFunction, BaseMetaModelTuner):

    def _propose(self, num_proposals, allow_duplicates):

        num_samples = num_proposals * 1000

        if allow_duplicates:
            proposed = self._sample(num_samples, allow_duplicates)

        elif self.tunable.SC > num_proposals:
            proposed = self._sample(num_samples, allow_duplicates)

        else:
            proposed = self._sample(self.tunable.SC, allow_duplicates)

        predicted = self._predict(proposed)
        index = self._acquire(predicted, num_proposals)
        return proposed[index]
