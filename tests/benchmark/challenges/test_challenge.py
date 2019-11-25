from unittest.mock import MagicMock

from btb.benchmark.challenges.challenge import Challenge


def test_get_tuner_params():

    # setup
    instance = MagicMock()

    # run
    result = Challenge.get_tuner_params(instance)

    # assert
    assert result == {}
