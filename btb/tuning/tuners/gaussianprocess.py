
from btb.tuning.acquisition.numpyargmax import NumpyArgMaxFunction
from btb.tuning.metamodels.gausianprocess import GaussianProcessMetaModel
from btb.tuning.tuners.base import BaseMetaModelTuner


class GaussianProcess(GaussianProcessMetaModel, NumpyArgMaxFunction, BaseMetaModelTuner):

    def _propose(self, num_proposals):
        if num_proposals == 1:
            return self.tunable.sample(1)

        best_candidates = list()

        for x in range(num_proposals):
            candidates = self.tunable.sample(num_proposals)
            predicted = self._predict(candidates)
            index = self._acquire(predicted)

            best_candidates.append(candidates[index])

        return best_candidates
