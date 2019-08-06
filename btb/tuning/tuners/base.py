# -*- coding: utf-8 -*-


class BaseTuner:

    def __init__(self, tunable, objectives):
        """Initialize the tuner by passing tunables."""
        pass

    def record(self, params, score):
        """Record the result of one trial.
        1. Encode as [0, 1]^p representation.
        2. Append to internal results store.
        3. Re-fit meta-model.
        """
        pass

    def propose(self, n):
        """Propose (one or more) new hyperparameter configuration(s).
        1. Create candidates.
        2. Use acquisition function to acquire candidates.
        3. Decode to irignal representation.
        """
        pass

    @property
    def best_score(self):
        """Get the best score achieved so far."""
        pass

    @property
    def best_params(self):
        """Get the hyperparameters that achieved the best score."""
        pass
