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

    # def test_accept_errors(self):
    #     tunables = {
    #         'a_tunable': {
    #             'a_parameter': {
    #                 'type': 'int',
    #                 'default': 0,
    #                 'range': [0, 2]
    #             }
    #         },
    #         'another_tunable': {
    #             'a_parameter': {
    #                 'type': 'int',
    #                 'default': 0,
    #                 'range': [0, 2]
    #             }
    #         }
    #     }

    #     def scorer(name, proposal):
    #         if name == 'another_tunable':
    #             raise Exception()
    #         else:
    #             return proposal['a_parameter']

    #     session = BTBSession(tunables, scorer)

    #     best = session.run(6)

    #     assert best['name'] == 'a_tunable'
    #     assert best['config'] == {'a_parameter': 2}
