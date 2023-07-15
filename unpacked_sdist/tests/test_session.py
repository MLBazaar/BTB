# -*- coding: utf-8 -*-

from collections import Counter, defaultdict
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np
from tqdm.auto import trange

from baytune.session import BTBSession
from baytune.tuning.tuners.base import StopTuning
from baytune.tuning.tuners.gaussian_process import GPTuner


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
        instance._maximize = False

        # run
        result = BTBSession._normalize(instance, 1)

        # assert
        assert result == -1

    def test___init__default(self):
        # run
        tunable = {"my_test_tuner": {"my_tunable_hp": {}}}
        scorer = "my_scorer"
        instance = BTBSession(tunable, scorer)

        # assert
        assert instance._tunables is tunable
        assert instance._scorer is scorer
        assert instance._tuner_class is GPTuner
        assert instance._best_normalized == -np.inf
        assert instance._normalized_scores == defaultdict(list)
        assert instance._tuners == {}
        assert instance._tunable_names == ["my_test_tuner"]
        assert instance._range is range
        assert instance._max_errors == 1
        assert instance._maximize

        assert instance.best_score is None
        assert instance.best_proposal is None
        assert instance.proposals == {}
        assert instance.iterations == 0
        assert instance.errors == Counter()

    def test___init__custom(self):
        # setup
        selector = MagicMock()

        # run
        tunable = {"my_test_tuner": {"my_tunable_hp": {}}}
        scorer = "my_scorer"
        instance = BTBSession(
            tunable,
            scorer,
            tuner_class="my_tuner",
            selector_class=selector,
            maximize=False,
            max_errors=2,
            verbose=True,
        )

        # assert
        assert instance._tunables is tunable
        assert instance._scorer is scorer
        assert instance._tuner_class == "my_tuner"
        assert instance._max_errors == 2
        assert instance._best_normalized == -np.inf
        assert instance._normalized_scores == defaultdict(list)
        assert instance._tuners == {}
        assert instance._tunable_names == ["my_test_tuner"]
        assert instance._range is trange

        assert instance.best_proposal is None
        assert instance.proposals == {}
        assert instance.iterations == 0
        assert instance.errors == Counter()

    def test__make_dumpable(self):
        # run
        randint = np.random.randint(1, dtype=np.integer)
        to_dump = {
            1: randint,
            "str": "None",
            "array": np.array([1, 2, 3]),
        }

        result = BTBSession._make_dumpable(MagicMock(), to_dump)

        # assert
        expected_result = {"1": int(randint), "str": None, "array": [1, 2, 3]}

        assert result == expected_result

    @patch("btb.session.np.random.choice")
    def test__get_next_tunable_name_normalized_scores(self, mock_np_random_choice):
        # setup
        mock_np_random_choice.return_value = "test_name"
        selector = MagicMock()
        selector.select.return_value = "test_name"

        instance = MagicMock(spec_set=BTBSession)
        instance._normalized_scores = [("test_name", 0.1), ("second_test_name", 0.2)]
        instance._selector = selector

        # run
        tunable_name = BTBSession._get_next_tunable_name(instance)

        # assert
        assert tunable_name == "test_name"
        mock_np_random_choice.assert_not_called()
        selector.select.assert_called_once_with(
            [
                ("test_name", 0.1),
                ("second_test_name", 0.2),
            ]
        )

    @patch("btb.session.np.random.choice")
    def test__get_next_tunable_name_normalized_scores_none(self, mock_np_random_choice):
        # setup
        mock_np_random_choice.return_value = "test_name"

        instance = MagicMock(spec_set=BTBSession)
        instance._normalized_scores = None
        instance._tunables = {
            "test_name": "test",
            "second_test_name": "second_test",
        }

        # python3.5 randomness issue, we read as in `btb.session`
        expected_mock_call = list(instance._tunables.keys())

        # run
        tunable_name = BTBSession._get_next_tunable_name(instance)

        # assert

        assert tunable_name == "test_name"
        mock_np_random_choice.assert_called_once_with(expected_mock_call)

    def test_propose_no_tunables(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance._tunables = None

        # run
        with self.assertRaises(StopTuning):
            BTBSession.propose(instance)

    @patch("btb.session.isinstance")
    @patch("btb.session.Tunable")
    def test_propose_normalized_scores_lt_tunable_names(
        self, mock_tunable, mock_isinstance
    ):
        # setup
        mock_tunable.from_dict.return_value.get_defaults.return_value = "defaults"
        mock_isinstance.return_value = True

        tuner = MagicMock()

        instance = MagicMock(spec_set=BTBSession)
        instance._tuner_class = tuner
        instance.proposals = {}
        instance._normalized_scores.__len__.return_value = 0
        instance._tunables = {"test_tunable": "test_spec"}
        instance._tunable_names = ["test_tunable"]

        instance._make_id.return_value = 1

        # run
        res_name, res_config = BTBSession.propose(instance)

        # assert
        assert res_name == "test_tunable"
        assert res_config == "defaults"

        expected_proposals = {
            1: {"id": 1, "name": "test_tunable", "config": "defaults"}
        }
        assert instance.proposals == expected_proposals

        instance._make_id.assert_called_once_with("test_tunable", "defaults")
        mock_tunable.from_dict.assert_called_once_with("test_spec")
        tuner.assert_called_once_with(mock_tunable.from_dict.return_value)
        mock_tunable.from_dict.return_value.get_defaults.assert_called_once_with()

        expected_isinstance_calls = [
            call("test_spec", dict),
            call("defaults", mock_tunable),
        ]
        mock_isinstance.has_calls(expected_isinstance_calls)

    def test_propose_normalized_scores_gt_tunable_names(self):
        # setup
        tuner = MagicMock()
        tuner.propose.return_value = "parameters"

        instance = MagicMock(spec_set=BTBSession)
        instance.proposals = {}
        instance._normalized_scores.__len__.return_value = 1

        instance._selector.select.return_value = "test_tunable"
        instance._tuners = {"test_tunable": tuner}
        instance._tunables = {"test_tunable": "test_spec"}
        instance._tunable_names = ["test_tunable"]
        instance._get_next_tunable_name.return_value = "test_tunable"

        instance._make_id.return_value = 1

        # run
        res_name, res_config = BTBSession.propose(instance)

        # assert
        assert res_name == "test_tunable"
        assert res_config == "parameters"

        expected_proposals = {
            1: {"id": 1, "name": "test_tunable", "config": "parameters"}
        }
        assert instance.proposals == expected_proposals

        instance._make_id.assert_called_once_with("test_tunable", "parameters")
        tuner.propose.assert_called_once_with(1)

    def test_propose_raise_error(self):
        # setup

        tuner = MagicMock()
        tuner.propose.side_effect = [StopTuning("test")]

        instance = MagicMock(spec_set=BTBSession)
        instance._normalized_scores.__len__.return_value = 1

        instance._selector.select.return_value = "test_tunable"
        instance._tuners = {"test_tunable": tuner}
        instance._tunables = {"test_tunable": "test_spec"}
        instance._tunable_names = ["test_tunable"]
        instance._get_next_tunable_name.return_value = "test_tunable"

        instance._make_id.return_value = 1

        # run
        with self.assertRaises(ValueError):
            BTBSession.propose(instance)

    @patch("btb.session.isinstance")
    @patch("btb.session.Tunable")
    def test_propose_tunable_cardinality_eq_one(self, mock_tunable, mock_isinstance):
        # setup
        mock_tunable.from_dict.return_value.cardinality = 1
        mock_tunable.from_dict.return_value.get_defaults.return_value = "parameters"
        mock_isinstance.return_value = True

        instance = MagicMock(spec_set=BTBSession)
        instance._tuners = {}
        instance._tunable_names = ["test_tunable"]
        instance.proposals = {}

        instance._make_id.return_value = 1

        # run
        tunable_name, config = BTBSession.propose(instance)

        # assert
        instance._make_id.assert_called_once_with("test_tunable", "parameters")
        instance._tuner_class.assert_not_called()

        assert instance._tuners == {"test_tunable": None}
        assert "test_tunable" == tunable_name
        assert "parameters" == config

    def test_propose_tuner_is_none(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance._tuners = {"test_tunable": None}
        instance._tunable_names = ["test_tunable"]
        instance._normalized_scores = None
        instance._get_next_tunable_name.return_value = "test_tunable"

        # run
        with self.assertRaises(ValueError):
            BTBSession.propose(instance)

    def test_handle_error_errors_lt_max_errors(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance.errors = Counter()
        instance._max_errors = 2

        # run
        BTBSession.handle_error(instance, "test")

        # assert
        instance._normalized_scores.pop.assert_not_called()
        instance._tunable_names.remove.assert_not_called()

    def test_handle_error_errors_gt_max_errors(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance.errors = Counter()
        instance._max_errors = 0

        # run
        BTBSession.handle_error(instance, "test")

        # assert
        instance._remove_tunable.assert_called_once_with("test")

    def test_record_score_is_none(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance._make_id.return_value = 0
        instance.proposals = [{"test": "test"}]
        instance.errors = Counter()
        instance._max_errors = 5

        # run
        BTBSession.record(instance, "test", "config", None)

        # assert
        instance.handle_error.assert_called_once_with("test")

    def test_record_score_gt_best(self):
        # setup
        tuner = MagicMock()

        instance = MagicMock(spec_set=BTBSession)
        instance._make_id.return_value = 0
        instance.proposals = [{"test": "test"}]
        instance._tuners = {"test": tuner}
        instance.best_proposal = None

        instance._best_normalized = 0
        instance._normalize.return_value = 1
        instance._normalized_scores = defaultdict(list)

        # run
        BTBSession.record(instance, "test", "config", 1)

        # assert
        expected_normalized_scores = defaultdict(list)
        expected_normalized_scores["test"].append(1)

        assert instance._normalized_scores == expected_normalized_scores
        assert instance.best_proposal == {"test": "test", "score": 1}
        assert instance._best_normalized == 1

        tuner.record.assert_called_once_with("config", 1)

    def test_record_score_gt_best_tuner_none(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance._make_id.return_value = 0
        instance.proposals = [{"test": "test"}]
        instance._tuners = {"test": None}
        instance.best_proposal = None

        instance._best_normalized = 0
        instance._normalize.return_value = 1
        instance._normalized_scores = defaultdict(list)

        # run
        BTBSession.record(instance, "test", "config", 1)

        # assert
        expected_normalized_scores = defaultdict(list)
        expected_normalized_scores["test"].append(1)

        assert instance._normalized_scores == expected_normalized_scores
        assert instance.best_proposal == {"test": "test", "score": 1}
        assert instance._best_normalized == 1

    def test_record_score_lt_best(self):
        # setup
        tuner = MagicMock()

        instance = MagicMock(spec_set=BTBSession)
        instance._make_id.return_value = 0
        instance.proposals = [{"test": "test"}]
        instance._tuners = {"test": tuner}
        instance.best_proposal = None

        instance._best_normalized = 10
        instance._normalize.return_value = 1
        instance._normalized_scores = defaultdict(list)

        # run
        BTBSession.record(instance, "test", "config", 1)

        # assert
        expected_normalized_scores = defaultdict(list)
        expected_normalized_scores["test"].append(1)

        assert instance.best_proposal is None
        assert instance._normalized_scores == expected_normalized_scores
        assert instance._best_normalized == 10

        tuner.record.assert_called_once_with("config", 1)

    def test_run_score(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance.propose.return_value = ("test", "config")
        instance._scorer.return_value = 1
        instance.best_proposal = {"test": "config"}
        instance._range = range
        instance.iterations = 0

        # run
        result = BTBSession.run(instance, 1)

        # assert
        instance._scorer.assert_called_once_with("test", "config")
        instance.record.assert_called_once_with("test", "config", 1)
        assert result == {"test": "config"}
        assert instance.iterations == 1

    def test_run_score_none(self):
        # setup
        instance = MagicMock(spec_set=BTBSession)
        instance.propose.return_value = ("test", {"hp": "test"})
        instance._scorer.side_effect = Exception()
        instance.best_proposal = {"test": "config"}
        instance._range = range
        instance.iterations = 0

        # run
        result = BTBSession.run(instance, 1)

        # assert
        instance._scorer.assert_called_once_with("test", {"hp": "test"})
        instance.record.assert_called_once_with("test", {"hp": "test"}, None)
        assert result == {"test": "config"}
        assert instance.iterations == 1
