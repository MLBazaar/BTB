# -*- coding: utf-8 -*-

from unittest import TestCase

import pytest

from btb.session import BTBSession
from btb.tuning import StopTuning


class BTBSessionTest(TestCase):

    @staticmethod
    def scorer(name, proposal):
        """score = name length + parameter.

        best proposal will be `a_tunable + a_parameter=0`
        """
        return len(name) + proposal['a_parameter']

    def test_stop(self):
        tunables = {
            'a_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 2]
                }
            }
        }

        session = BTBSession(tunables, self.scorer)

        with pytest.raises(StopTuning):
            session.run()

    def test_maximize(self):
        tunables = {
            'a_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 2]
                }
            }
        }

        session = BTBSession(tunables, self.scorer)

        best = session.run(3)

        assert best == session.best_proposal

        assert best['name'] == 'a_tunable'
        assert best['config'] == {'a_parameter': 2}

    def test_minimize(self):
        tunables = {
            'a_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 2]
                }
            }
        }

        session = BTBSession(tunables, self.scorer, maximize=False)

        best = session.run(3)

        assert best == session.best_proposal
        assert best['name'] == 'a_tunable'
        assert best['config'] == {'a_parameter': 0}

    def test_multiple(self):
        tunables = {
            'a_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 2]
                }
            },
            'another_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 2]
                }
            }
        }

        session = BTBSession(tunables, self.scorer)

        best = session.run(6)

        assert best['name'] == 'another_tunable'
        assert best['config'] == {'a_parameter': 2}

    def test_errors(self):
        tunables = {
            'a_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 2]
                }
            },
            'another_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 2]
                }
            }
        }

        def scorer(name, proposal):
            if name == 'another_tunable':
                raise Exception()
            else:
                return proposal['a_parameter']

        session = BTBSession(tunables, scorer)

        best = session.run(4)

        assert best['name'] == 'a_tunable'
        assert best['config'] == {'a_parameter': 2}

    def test_normalized_score_becomes_none(self):
        """Tunables that worked at some point but end up removed are not tried again.

        After commit ``6a08dc3cf1b68b35630cae6a87783aec4e2c9f83`` the following
        scenario has been observed:

        - One tunable produces a score at least once and then fails the next trials.
        - All the other tunables never produce any score.
        - Once all the tuners are created, only the one that produced a score is used.
        - After enough errors, this one is discarded, so `_normalized_errors` is empty.
        - Since a random.choice is used over the list of tunables, which still contains
          the one tha has been discarded, at some point the discarded one is tried again.

        This test certifies that this scenario cannot happen again, by validating that
        the number of errors is always ``max_errors`` at most.
        """
        scores = []

        def scorer(name, proposal):
            """Produce a score for the first trial and then fail forever."""
            if not scores:
                scores.append(1)   # boolean variable fails due to scope unles using global
                return 1

            raise Exception()

        tunables = {
            'a_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 10]
                }
            },
            'another_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 10]
                }
            }
        }

        session = BTBSession(tunables, scorer, max_errors=3)

        with pytest.raises(StopTuning):
            session.run(8)

        assert session.errors == {'a_tunable': 3, 'another_tunable': 3}

    @pytest.mark.skip(reason="This is not implemented yet")
    def test_allow_duplicates(self):
        tunables = {
            'a_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 2]
                }
            }
        }

        session = BTBSession(tunables, self.scorer, allow_duplicates=True)

        best = session.run(10)

        assert best['name'] == 'another_tunable'
        assert best['config'] == {'a_parameter': 2}

    def test_allow_errors(self):
        tunables = {
            'a_tunable': {
                'a_parameter': {
                    'type': 'int',
                    'default': 0,
                    'range': [0, 1]
                }
            }
        }

        def scorer(name, proposal):
            if proposal['a_parameter'] == 0:
                raise Exception()

            return 1

        session = BTBSession(tunables, scorer, max_errors=10)

        best = session.run(10)

        assert best['name'] == 'a_tunable'
        assert best['config'] == {'a_parameter': 1}
