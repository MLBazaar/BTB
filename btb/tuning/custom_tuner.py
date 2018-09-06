from btb.tuning.tuner import BaseTuner


class CustomTuner(BaseTuner):
    """Bare-bones tuner that returns a random set of parameters each time."""

    def propose(self):
        """Generate and return a random set of parameters."""
        return self._create_candidates(1)[0, :]
