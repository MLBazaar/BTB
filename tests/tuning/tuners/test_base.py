# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np

from btb.tuning.tunable import Tunable
from btb.tuning.tuners.base import BaseMetaModelTuner, BaseTuner, StopTuning


class TestBaseTuner(TestCase):
    """Test BaseTuner class."""

    def test___init__defaults(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)

        # run
        instance = BaseTuner(tunable)

        # assert
        assert instance.tunable is tunable
        assert isinstance(instance.trials, np.ndarray)
        assert isinstance(instance.raw_scores, np.ndarray)
        assert isinstance(instance.scores, np.ndarray)
        assert isinstance(instance._trials_set, set)
        assert isinstance(instance.maximize, bool)

        assert instance.maximize
        assert instance.trials.shape == (0, 1)
        assert instance.raw_scores.shape == (0, 1)
        assert instance.trials.dtype == np.float
        assert instance.raw_scores.dtype == np.float

    def test___init__maximize_false(self):
        # setup
        tunable = MagicMock(spec_set=Tunable)

        # run
        instance = BaseTuner(tunable, False)

        # assert
        assert isinstance(instance.tunable, MagicMock)
        assert isinstance(instance.trials, np.ndarray)
        assert isinstance(instance.raw_scores, np.ndarray)
        assert isinstance(instance.scores, np.ndarray)
        assert isinstance(instance._trials_set, set)
        assert isinstance(instance.maximize, bool)

        assert not instance.maximize

    def test__check_proposals_proposals_gt_cardinality(self):
        """Test that ``StopTuning`` is being raised if ``proposals`` is greater than
        ``self.tunable.cardinality``.
        """
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.cardinality = 4

        # run / assert
        with self.assertRaises(StopTuning):
            BaseTuner._check_proposals(instance, 5)

    def test__check_proposals_trials_eq_cardinality(self):
        """Test that ``StopTuning`` is being raised if ``self.trials`` is equal to
        ``self.tunable.cardinality``.
        """
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.cardinality = 2
        instance._trials_set.__len__.return_value = 2

        # run / assert
        with self.assertRaises(StopTuning):
            BaseTuner._check_proposals(instance, 1)

    def test__check_proposals_trials_and_proposals_gt_cardinality(self):
        """Test that ``StopTuning`` is being raised if ``proposals`` + ``len(self.trials)``
        is greater than ``self.tunable.cardinality``.
        """
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.cardinality = 4
        instance._trials_set.__len__.return_value = 2

        # run / assert
        with self.assertRaises(StopTuning):
            BaseTuner._check_proposals(instance, 3)

    def test__check_proposals_not_raise(self):
        """Test that ``StopTuning`` is not being raised."""
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.cardinality = 4
        instance._trials_set.__len__.return_value = 2

        # run / assert
        result = BaseTuner._check_proposals(instance, 1)
        assert result is None

    def test_propose_one_value_no_duplicates(self):
        """Test that ``propose`` method calls it's child implemented method with
        ``allow_duplicates`` as ``False``.
        """
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        inverse_return = instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1]
        instance._propose = MagicMock(return_value=1)

        # run
        result = BaseTuner.propose(instance, 1)

        # assert
        instance._check_proposals.assert_called_once_with(1)
        instance._propose.assert_called_once_with(1, False)
        instance.tunable.inverse_transform.called_once_with(1)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == 1

    @patch('btb.tuning.tuners.base.BaseTuner._check_proposals')
    def test_propose_one_value_allow_duplicates(self, mock__check_proposals):
        """Test that ``propose`` method calls it's child implemented method with
        ``allow_duplicates`` as ``True``.
        """
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        inverse_return = instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1]
        instance._propose = MagicMock(return_value=1)

        # run
        result = BaseTuner.propose(instance, 1, allow_duplicates=True)

        # assert
        instance._check_proposals.assert_not_called()
        instance._propose.assert_called_once_with(1, True)
        instance.tunable.inverse_transform.called_once_with(1)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == 1

    @patch('btb.tuning.tuners.base.BaseTuner._check_proposals')
    def test_propose_many_values_no_duplicates(self, mock__check_proposals):
        """Test that ``propose`` method calls it's child implemented method with more than one
        proposals and ``allow_duplicates`` as ``False``.
        """
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        inverse_return = instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1, 2]
        instance._propose = MagicMock(return_value=2)

        # run
        result = BaseTuner.propose(instance, 2)

        # assert
        instance._propose.assert_called_once_with(2, False)
        instance.tunable.inverse_transform.called_once_with(2)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == [1, 2]

    @patch('btb.tuning.tuners.base.BaseTuner._check_proposals')
    def test_propose_many_values_allow_duplicates(self, mock__check_proposals):
        """Test that ``propose`` method calls it's child implemented method with more than one
        proposals and ``allow_duplicates`` as ``True``.
        """

        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        inverse_return = instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1, 2]
        instance._propose = MagicMock(return_value=2)

        # run
        result = BaseTuner.propose(instance, 2, allow_duplicates=True)

        # assert
        instance._propose.assert_called_once_with(2, True)
        instance.tunable.inverse_transform.called_once_with(2)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == [1, 2]

    def test__sample_allow_duplicates(self):
        """Test the method ``_sample``when using duplicates."""
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.sample.return_value = 1

        # run
        result = BaseTuner._sample(instance, 1, True)

        # assert
        instance.tunable.sample.assert_called_once_with(1)
        assert result == 1

    def test__sample_not_allow_duplicates(self):
        """Test that the method ``_sample`` returns ``np.ndarray`` when not using duplicates."""
        # setup
        instance = MagicMock()
        instance._trials_set = set()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.sample.return_value = np.array([[3]])

        # run
        result = BaseTuner._sample(instance, 1, False)

        # assert
        instance.tunable.sample.assert_called_once_with(1)
        np.testing.assert_array_equal(result, np.array([[3]]))

    def test_sample_no_duplicates_more_than_one_loop(self):
        """Test that the method ``_sample`` returns ``np.ndarray`` when not using duplicates and
        perfroms more than one iteration.
        """
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance._trials_set = set({(1, ), (2, )})

        side_effect = [np.array([[3]]), np.array([[1]]), np.array([[1]]), np.array([[4]])]
        instance.tunable.sample.side_effect = side_effect

        # run
        result = BaseTuner._sample(instance, 2, False)

        # assert
        assert instance.tunable.sample.call_args_list == [call(2), call(2), call(2), call(2)]
        np.testing.assert_array_equal(result, np.array([[3], [4]]))

    def test_record_list_maximize_true(self):
        """Test that the method record updates the ``trials``  and ``scores``."""

        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.transform.return_value = np.array([[1, 0]])
        instance.trials = np.empty((0, 2), dtype=np.float)
        instance._trials_set = set()
        instance.scores = None
        instance.maximize = True
        instance.raw_scores = np.empty((0, 1), dtype=np.float)

        # run
        BaseTuner.record(instance, [1], [0.1])

        # assert
        instance.tunable.transform.assert_called_once_with([1])

        np.testing.assert_array_equal(instance.trials, np.array([[1, 0]]))
        assert instance._trials_set == set({(1, 0)})
        np.testing.assert_array_equal(instance.raw_scores, np.array([0.1]))
        np.testing.assert_array_equal(instance.scores, np.array([0.1]))

    def test_record_list_maximize_false(self):
        """Test that the method record updates the ``trials``  and ``scores``."""

        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.transform.return_value = np.array([[1, 0]])
        instance.trials = np.empty((0, 2), dtype=np.float)
        instance._trials_set = set()
        instance.scores = None
        instance.maximize = False
        instance.raw_scores = np.empty((0, 1), dtype=np.float)

        # run
        BaseTuner.record(instance, [1], [0.1])

        # assert
        instance.tunable.transform.assert_called_once_with([1])

        np.testing.assert_array_equal(instance.trials, np.array([[1, 0]]))
        assert instance._trials_set == set({(1, 0)})
        np.testing.assert_array_equal(instance.raw_scores, np.array([0.1]))
        np.testing.assert_array_equal(instance.scores, np.array([-0.1]))

    def test_record_scalar_values(self):
        """Test that the method record performs an update to ``trials`` and ``scores`` when called
        with a scalar value.
        """
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.trials = np.empty((0, 2), dtype=np.float)
        instance.raw_scores = np.empty((0, 1), dtype=np.float)
        instance._trials_set = set()
        instance.tunable.transform.return_value = np.array([[1, 0]])

        # run
        BaseTuner.record(instance, 1, 0.1)

        # assert
        instance.tunable.transform.assert_called_once_with(1)
        np.testing.assert_array_equal(instance.trials, np.array([[1, 0]]))
        assert instance._trials_set == set({(1, 0)})
        np.testing.assert_array_equal(instance.raw_scores, np.array([0.1]))
        np.testing.assert_array_equal(instance.scores, np.array([0.1]))

    def test_record_raise_error(self):
        """Test that the method record raises a ``ValueError`` when ``len(trials)`` is different
        to ``len(scores)``.
        """
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.transform.return_value = np.array([[1, 0]])

        # run / assert
        with self.assertRaises(ValueError):
            BaseTuner.record(instance, 1, [1, 2])


class TestBaseMetaModelTuner(TestCase):
    """Test BaseMetaModelTuner class."""

    @patch('btb.tuning.tuners.base.super')
    def test___init___default_values(self, mock_super):
        # setup
        tunable = MagicMock(spec_set=Tunable)
        instance = MagicMock()
        instance.__init_metamodel__ = MagicMock()
        instance.__init_acquisition__ = MagicMock()

        # run
        BaseMetaModelTuner.__init__(instance, tunable)

        # assert
        assert instance.num_candidates == 1000
        assert instance.min_trials == 2
        instance.__init_metamodel__.assert_called_once_with()
        instance.__init_acquisition__.assert_called_once_with()

    @patch('btb.tuning.tuners.base.super')
    def test___init___users_values(self, mock_super):

        # setup
        tunable = MagicMock(spec_set=Tunable)
        instance = MagicMock()
        instance._metamodel_kwargs = {'a': 'test'}
        instance._acquisition_kwargs = {'a': 'acquisition_test'}
        instance.__init_metamodel__ = MagicMock()
        instance.__init_acquisition__ = MagicMock()

        # run
        BaseMetaModelTuner.__init__(
            instance,
            tunable,
            maximize=False,
            num_candidates=5,
            min_trials=20,
        )

        # assert
        assert instance.num_candidates == 5
        assert instance.min_trials == 20
        instance.__init_metamodel__.assert_called_once_with(a='test')
        instance.__init_acquisition__.assert_called_once_with(a='acquisition_test')

    def test__proposemin_trials_gt__trials_set(self):
        # setup
        instance = MagicMock()
        instance.min_trials = 1
        instance._trials_set.__len__.return_value = 0
        instance._sample.return_value = 'sample'

        # run
        result = BaseMetaModelTuner._propose(instance, 1, True)

        # assert
        instance._sample.assert_called_once_with(1, True)
        assert result == 'sample'

    def test__proposemin_trials_lt__trials_set_allow_duplicates(self):
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.cardinality = 3
        instance.min_trials = 0
        instance.num_candidates = 10
        instance._trials_set.__len__.return_value = 1
        instance._sample.return_value = np.array([1])
        instance._predict.return_value = 'predicted'
        instance._acquire.return_value = 0

        # run
        result = BaseMetaModelTuner._propose(instance, 1, True)

        # assert
        instance._sample.assert_called_once_with(10, True)
        instance._predict.assert_called_once_with(np.array([1]))
        assert result == 1

    def test__proposemin_trials_lt__trials_set_not_allow_duplicates(self):
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.cardinality = 3
        instance.min_trials = 0
        instance.num_candidates = 10
        instance._trials_set.__len__.return_value = 1
        instance._sample.return_value = np.array([1])
        instance._predict.return_value = 'predicted'
        instance._acquire.return_value = 0

        # run
        result = BaseMetaModelTuner._propose(instance, 1, False)

        # assert
        instance._sample.assert_called_once_with(2, False)
        instance._predict.assert_called_once_with(np.array([1]))
        instance._acquire.assert_called_once_with('predicted', 1)
        assert result == 1

    def test__proposemin_trials_lt__trials_set_not_allow_duplicates_num_samples_is_min(self):
        # setup
        instance = MagicMock()
        instance.tunable = MagicMock(spec_set=Tunable)
        instance.tunable.cardinality = 3
        instance.min_trials = 0
        instance.num_candidates = 0
        instance._trials_set.__len__.return_value = 1
        instance._sample.return_value = np.array([1])
        instance._predict.return_value = 'predicted'
        instance._acquire.return_value = 0

        # run
        result = BaseMetaModelTuner._propose(instance, 1, False)

        # assert
        instance._sample.assert_called_once_with(0, False)
        instance._predict.assert_called_once_with(np.array([1]))
        instance._acquire.assert_called_once_with('predicted', 1)
        assert result == 1

    @patch('btb.tuning.tuners.base.super')
    def test_record(self, mock_super):
        # setup
        instance = MagicMock()
        instance.trials = np.array([2])
        instance.scores = 1
        instance.min_trials = 1

        # run
        BaseMetaModelTuner.record(instance, 1, 1)

        # assert
        mock_super.return_value.record.assert_called_once_with(1, 1)
        instance._fit.assert_called_once_with(np.array([2]), 1)
