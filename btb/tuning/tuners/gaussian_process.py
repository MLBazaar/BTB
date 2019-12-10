# -*- coding: utf-8 -*-

"""Package where the tuners based on GaussianProcessMetaModel are defined."""

from btb.tuning.acquisition.expected_improvement import ExpectedImprovementAcquisition
from btb.tuning.acquisition.predicted_score import PredictedScoreAcquisition
from btb.tuning.metamodels.gaussian_process import GaussianProcessMetaModel
from btb.tuning.tuners.base import BaseMetaModelTuner


class GPTuner(GaussianProcessMetaModel, PredictedScoreAcquisition, BaseMetaModelTuner):
    """Gaussian Process Tuner.

    This class uses a ``GaussianProcessRegressor`` model from the ``sklearn.gaussian_process``
    package, using a ``numpy.argmax`` function to return the better configurations predicted
    from the model.
    """
    def __init__(self, tunable, maximize=True, num_candidates=1000,
                 min_trials=2, length_scale=0.1):
        """Create an instance of ``GPTuner``.

        Args:
            tunable (btb.tuning.tunable.Tunable):
                Instance of a tunable class containing hyperparameters to be tuned.
            num_candidates (int):
                Number of samples to generate and select the best of it for each proposal.
                Defaults to 1000.
            maximize (bool):
                If ``True`` the model will understand that the score bigger is better, if ``False``
                the smaller is better. Defaults to ``True``.
            min_trials (int):
                Number of recorded ``trials`` needed to perform a fitting over the model.
                Defaults to 2.
            length_scale (float or array):
                A float or array with shape ``(n_features,)``, used for the default ``RBF`` kernel.
        """
        self._metamodel_kwargs = {'length_scale': length_scale}
        super().__init__(tunable, maximize, num_candidates, min_trials)

    def __repr__(self):
        length_scale = self._metamodel_kwargs.get('length_scale')
        args = (self.tunable, self.maximize, self.num_candidates, self.min_trials, length_scale)
        return ('GPTuner(tunable={}, maximize={},'
                'num_candidates={}, min_trials={},'
                'length_scale={})').format(*args)


class GPEiTuner(GaussianProcessMetaModel, ExpectedImprovementAcquisition, BaseMetaModelTuner):
    """Gaussian Process Expected Improvement Tuner.

    This class uses a ``GaussianProcessRegressor`` model from the ``sklearn.gaussian_process``
    package, using an ``ExpectedImprovement`` function to return the better configurations
    predicted from the model.
    """
    def __init__(self, tunable, maximize=True, num_candidates=1000,
                 min_trials=2, length_scale=0.1):
        """Create an instance of ``GPEiTuner``.

        Args:
            tunable (btb.tuning.tunable.Tunable):
                Instance of a tunable class containing hyperparameters to be tuned.
            num_candidates (int):
                Number of samples to generate and select the best of it for each proposal.
                Defaults to 1000.
            maximize (bool):
                If ``True`` the model will understand that the score bigger is better, if ``False``
                the smaller is better. Defaults to ``True``.
            min_trials (int):
                Number of recorded ``trials`` needed to perform a fitting over the model.
                Defaults to 2.
            length_scale (float or array):
                A float or array with shape ``(n_features,)``, used for the default ``RBF`` kernel.
        """
        self.length_scale = length_scale
        self._metamodel_kwargs = {'length_scale': self.length_scale}
        super().__init__(tunable, maximize, num_candidates, min_trials)

    def __repr__(self):
        length_scale = self._metamodel_kwargs.get('length_scale')
        args = (self.tunable, self.maximize, self.num_candidates, self.min_trials, length_scale)
        return ('GPTuner(tunable={}, maximize={},'
                'num_candidates={}, min_trials={},'
                'length_scale={})').format(*args)
