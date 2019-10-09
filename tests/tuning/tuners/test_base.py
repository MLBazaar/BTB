# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np

from btb.tuning.tuners.base import BaseTuner


class TestBaseTuner(TestCase):
    """Test BaseTuner class."""

    def test___init__(self):
        # setup
        tunable = MagicMock()
        tunable.dimensions = 2

        # run
        instance = BaseTuner(tunable)

        # assert
        assert isinstance(instance.tunable, MagicMock)
        assert isinstance(instance.trials, np.ndarray)
        assert isinstance(instance.scores, np.ndarray)

        assert instance.trials.shape == (0, 2)
        assert instance.scores.shape == (0, 1)
        assert instance.trials.dtype == np.float
        assert instance.scores.dtype == np.float

    def test__check_proposals_proposals_gt_cardinality(self):
        """Test that ``ValueError`` is being raised if ``proposals`` is greater than
        ``self.tunable.cardinality``.
        """
        # setup
        instance = MagicMock()
        instance.tunable.cardinality = 4

        # run / assert
        with self.assertRaises(ValueError):
            BaseTuner._check_proposals(instance, 5)

    def test__check_proposals_trials_eq_cardinality(self):
        """Test that ``ValueError`` is being raised if ``self.trials`` is equal to
        ``self.tunable.cardinality``.
        """
        # setup
        instance = MagicMock()
        instance.tunable.cardinality = 2
        instance.trials = np.array([[1], [2]])

        # run / assert
        with self.assertRaises(ValueError):
            BaseTuner._check_proposals(instance, 1)

    def test__check_proposals_trials_and_proposals_gt_cardinality(self):
        """Test that ``ValueError`` is being raised if ``proposals`` + ``len(self.trials)``
        is greater than ``self.tunable.cardinality``.
        """
        # setup
        instance = MagicMock()
        instance.tunable.cardinality = 4
        instance.trials = np.array([[1], [2]])

        # run / assert
        with self.assertRaises(ValueError):
            BaseTuner._check_proposals(instance, 3)

    def test__check_proposals_not_raise(self):
        """Test that ``ValueError`` is not being raised."""
        # setup
        instance = MagicMock()
        instance.tunable.cardinality = 4
        instance.trials = np.array([[1], [2]])

        # run / assert
        result = BaseTuner._check_proposals(instance, 1)
        assert result is None

    def test_propose_one_value_no_duplicates(self):
        """Test that ``propose`` method calls it's child implemented method with
        ``allow_duplicates`` as ``False``.
        """
        # setup
        instance = MagicMock()
        inverse_return = instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1]
        instance._propose = MagicMock(return_value=1)

        # run
        result = BaseTuner.propose(instance, 1)

        # assert
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
        inverse_return = instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1]
        instance._propose = MagicMock(return_value=1)

        # run
        result = BaseTuner.propose(instance, 1, allow_duplicates=True)

        # assert
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
        instance.trials = np.array([[1], [2]])
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
        instance.trials = np.array([[1], [2]])

        side_effect = [np.array([[3]]), np.array([[1]]), np.array([[1]]), np.array([[4]])]
        instance.tunable.sample.side_effect = side_effect

        # run
        result = BaseTuner._sample(instance, 2, False)

        # assert
        assert instance.tunable.sample.call_args_list == [call(2), call(2), call(2), call(2)]
        np.testing.assert_array_equal(result, np.array([[3], [4]]))

    def test_record_list(self):
        """Test that the method record updates the ``trials``  and ``scores``."""

        # setup
        instance = MagicMock()
        instance.tunable.transform.return_value = np.array([[1, 0]])
        instance.trials = np.empty((0, 2), dtype=np.float)
        instance.scores = np.empty((0, 1), dtype=np.float)

        # run
        BaseTuner.record(instance, [1], [0.1])

        # assert
        instance.tunable.transform.assert_called_once_with([1])

        np.testing.assert_array_equal(instance.trials, np.array([[1, 0]]))
        np.testing.assert_array_equal(instance.scores, np.array([0.1]))

    def test_record_scalar_values(self):
        """Test that the method record performs an update to ``trials`` and ``scores`` when called
        with a scalar value.
        """
        # setup
        instance = MagicMock()
        instance.trials = np.empty((0, 2), dtype=np.float)
        instance.scores = np.empty((0, 1), dtype=np.float)
        instance.tunable.transform.return_value = np.array([[1, 0]])

        # run
        BaseTuner.record(instance, 1, 0.1)

        # assert
        instance.tunable.transform.assert_called_once_with(1)
        np.testing.assert_array_equal(instance.trials, np.array([[1, 0]]))
        np.testing.assert_array_equal(instance.scores, np.array([0.1]))

    def test_record_raise_error(self):
        """Test that the method record raises a ``ValueError`` when ``len(trials)`` is different
        to ``len(scores)``.
        """
        # setup
        instance = MagicMock()
        instance.tunable.transform.return_value = np.array([[1, 0]])

        # run / assert
        with self.assertRaises(ValueError):
            BaseTuner.record(instance, 1, [1, 2])
