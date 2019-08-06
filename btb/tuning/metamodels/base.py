# -*- coding: utf-8 -*-


class MetaModel:

    def __init__(self, *args, **kwargs):
        """Enables cooperative multiple inheritance."""
        super().__init__(*args, **kwargs)
        self._init_model()

    def _init_model(self):
        self._model = None

    def _fit(self, params, scores):
        """Process params and scores and fit internal meta-model.
        Args:
            params (ArrayLike): 2d array-like with shape (n_trials, n_params)
            scores (ArrayLike): 2d array-like with shape (n_trials, 1)
        """
        pass

    def _predict(self, candidates):
        """Predict performance for candidate params under this meta-model.
        Depending on the meta-model, the predictions could be point predictions or could also
        include a standard deviation at that point (like with a Gaussian Process meta-model).

        Args:
            candidates (ArrayLike): 2D array-like with shape (n_cadidates, n_params)

        Returns:
            predictions (ArrayLike): 2D array-like with shape (n_candidates, n_outputs)
        """
        pass
