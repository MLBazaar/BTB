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
        """
        Due to a bug in commit ``@531990e``, we create this test to avoid the following problem:

        ``BTBSession`` was runing with a ``max_errors=5``, with two tunables. One of them doesn't
        generate any score and the other one generates scores until it starts failing.
        When the first one didn't generate any score for ``5`` iterations, it gets removed and
        only the one that generated scores is left.

        This one stops generating new scores and after ``5`` errors should be removed and not used
        anymore. However, this one gets proposed atleast one more time, when
        ``self._normalized_scores`` becomes ``None`` this one is being returned, and also there
        was a posibility that the first one gets returned aswell as ``numpy.random`` was being
        called with the wrong dictionary keys: ``self._tuners.keys()`` instead of
        ``self._tunables.keys()``.

        As this test doesn't raise any exception, it was detected thro the log. We are using
        ``session.iterations`` to ensure that after the ``max_errors`` for both tunables is
        reached, the ``session`` ends and doesn't continue like in the commit above.
        """
        proposals = [1, 1, 0, 0, 0]

        def scorer(name, proposal):
            if name == 'another_tunable':
                raise Exception()

            else:
                score = proposals.pop(0)
                if score == 1:
                    return score
                else:
                    raise Exception()

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

        session = BTBSession(tunables, scorer, max_errors=3)

        with pytest.raises(StopTuning):
            session.run(22)

        # 9 is 3 for another_tunable, 5 for a_tunable, +1 from run before StopTuning is raised.
        assert session.iterations == 9

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
