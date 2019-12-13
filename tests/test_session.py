# -*- coding: utf-8 -*-

from collections import Counter, defaultdict
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from tqdm.autonotebook import trange

from btb.session import BTBSession
from btb.tuning.tuners.gaussian_process import GPTuner


class TestBTBSession(TestCase):

    def test__normalize_maximize_true(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)

        # run
        result = BTBSession._normalize(instance, 1)

        # assert
        assert result == 1

    def test__normalize_maximize_false(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance.maximize = False

        # run
        result = BTBSession._normalize(instance, 1)

        # assert
        assert result == -1

    def test___init__default(self):
        # run
        tunable = {'my_test_tuner': {'my_tunable_hp': {}}}
        scorer = 'my_scorer'
        instance = BTBSession(tunable, scorer)
        # assert
        assert instance.tunables is tunable
        assert instance.scorer is scorer
        assert instance.tuner is GPTuner
        assert instance.best_proposal is None
        assert instance.proposals == {}
        assert instance.iterations == 0
        assert instance.errors == Counter()
        assert instance.max_errors == 1
        assert instance._best_normalized == -np.inf
        assert instance._normalized_scores == defaultdict(list)
        assert instance._tuners == {}
        assert instance._tunable_names == ['my_test_tuner']
        assert instance._range is range

    def test___init__custom(self):
        # setup
        selector = MagicMock()

        # run
        tunable = {'my_test_tuner': {'my_tunable_hp': {}}}
        scorer = 'my_scorer'
        instance = BTBSession(
            tunable,
            scorer,
            tuner='my_tuner',
            selector=selector,
            maximize=False,
            max_errors=2,
            verbose=True
        )

        # assert
        assert instance.tunables is tunable
        assert instance.scorer is scorer
        assert instance.tuner == 'my_tuner'
        assert instance.best_proposal is None
        assert instance.proposals == {}
        assert instance.iterations == 0
        assert instance.errors == Counter()
        assert instance.max_errors == 2
        assert instance._best_normalized == np.inf
        assert instance._normalized_scores == defaultdict(list)
        assert instance._tuners == {}
        assert instance._tunable_names == ['my_test_tuner']
        assert instance._range is trange

    def test__make_dumpable(self):
        # run
        randint = np.random.randint(1, dtype=np.integer)
        to_dump = {
            1: randint,
            'str': 'None',
            'array': np.array([1, 2, 3]),
        }

        result = BTBSession._make_dumpable(MagicMock(), to_dump)

        # assert
        expected_result = {
            '1': int(randint),
            'str': None,
            'array': [1, 2, 3]
        }

        assert result == expected_result

    def test_propose_no_tunables(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance.tunables = None

        # run
        with self.assertRaises(ValueError):
            BTBSession.propose(instance)

    @patch('btb.session.Tunable')
    def test_propose_normalized_scores_lt_tunable_names(self, mock_tunable):
        # setup
        mock_tunable.from_dict.return_value.get_defaults.return_value = 'defaults'

        tuner = MagicMock()

        instance = MagicMock(spec_set=BTBSession)
        instance.tuner = tuner
        instance.proposals = {}
        instance._normalized_scores.__len__.return_value = 0
        instance.tunables = {'test_tunable': 'test_spec'}
        instance._tunable_names = ['test_tunable']

        instance._make_id.return_value = 1

        # run
        res_name, res_config = BTBSession.propose(instance)

        # assert
        assert res_name == 'test_tunable'
        assert res_config == 'defaults'

        expected_proposals = {
            1: {
                'id': 1,
                'name': 'test_tunable',
                'config': 'defaults'
            }
        }
        assert instance.proposals == expected_proposals

        instance._make_id.assert_called_once_with('test_tunable', 'defaults')
        mock_tunable.from_dict.assert_called_once_with('test_spec')
        tuner.assert_called_once_with(mock_tunable.from_dict.return_value)
        mock_tunable.from_dict.return_value.get_defaults.assert_called_once_with()

    def test_propose_normalized_scores_gt_tunable_names(self):
        # setup
        tuner = MagicMock()
        tuner.propose.return_value = 'parameters'

        instance = MagicMock(spec_set=BTBSession)
        instance.proposals = {}
        instance._normalized_scores.__len__.return_value = 1

        instance.selector.select.return_value = 'test_tunable'
        instance._tuners = {'test_tunable': tuner}
        instance.tunables = {'test_tunable': 'test_spec'}
        instance._tunable_names = ['test_tunable']

        instance._make_id.return_value = 1

        # run
        res_name, res_config = BTBSession.propose(instance)

        # assert
        assert res_name == 'test_tunable'
        assert res_config == 'parameters'

        expected_proposals = {
            1: {
                'id': 1,
                'name': 'test_tunable',
                'config': 'parameters'
            }
        }
        assert instance.proposals == expected_proposals

        instance._make_id.assert_called_once_with('test_tunable', 'parameters')
        tuner.propose.assert_called_once_with(1)

    def test_propose_raise_error(self):
        # setup

        tuner = MagicMock()
        tuner.propose.side_effect = ValueError('test')

        instance = MagicMock(spec_set=BTBSession)
        instance._normalized_scores.__len__.return_value = 1

        instance.selector.select.return_value = 'test_tunable'
        instance._tuners = {'test_tunable': tuner}
        instance.tunables = {'test_tunable': 'test_spec'}
        instance._tunable_names = ['test_tunable']

        instance._make_id.return_value = 1

        # run
        result = BTBSession.propose(instance)

        # assert
        assert result is None
        tuner.propose.assert_called_once_with(1)

    def test_record_score_is_none_errors_lt_max_errors(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance._make_id.return_value = 0
        instance.proposals = [{'test': 'test'}]
        instance.errors = Counter()
        instance.max_errors = 5

        # run
        BTBSession.record(instance, 'test', 'config', None)

        # assert
        instance._normalized_scores.pop.assert_not_called()
        instance._tunable_names.remove.assert_not_called()

    def test_record_score_is_none_errors_gt_max_errors(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance._make_id.return_value = 0
        instance.proposals = [{'test': 'test'}]
        instance.errors = Counter()
        instance.max_errors = 0

        # run
        BTBSession.record(instance, 'test', 'config', None)

        # assert
        instance._normalized_scores.pop.assert_called_once_with('test', None)
        instance._tunable_names.remove.assert_called_once_with('test')

    def test_record_score_score_gt_best(self):
        # setup
        tuner = MagicMock()

        instance = MagicMock(spec_set=BTBSession)
        instance._make_id.return_value = 0
        instance.proposals = [{'test': 'test'}]
        instance._tuners = {'test': tuner}

        instance._best_normalized = 0
        instance._normalize.return_value = 1

        # run
        BTBSession.record(instance, 'test', 'config', 1)

        # assert
        tuner.record.assert_called_once_with('config', 1)

        instance.best_proposal == {'test': 'test', 'score': 1}
