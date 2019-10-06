# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np

from btb.tuning.tuners.base import BaseTuner


class TestBaseTuner(TestCase):
    """Test BaseTuner class."""

    class Tuner(BaseTuner):
        """Simple BaseTuner child."""
        def _propose(self, num_proposals, allow_duplicates):
            return num_proposals

    def setUp(self):
        tunable = MagicMock()
        tunable.K = 2
        self.instance = self.Tuner(tunable)

    def test___init__(self):
        assert isinstance(self.instance.tunable, MagicMock)
        assert isinstance(self.instance.trials, np.ndarray)
        assert isinstance(self.instance.scores, np.ndarray)

        assert self.instance.trials.shape == (0, 2)
        assert self.instance.scores.shape == (0, 1)
        assert self.instance.trials.dtype == np.float
        assert self.instance.scores.dtype == np.float

    def test__check_proposals_proposals_gt_sc(self):
        """Test that ``ValueError`` is being raised if ``proposals`` is greater than
        ``self.tunable.SC``.
        """
        # setup
        self.instance.tunable.SC = 4

        # run / assert
        with self.assertRaises(ValueError):
            self.instance._check_proposals(5)

    def test__check_proposals_trials_eq_sc(self):
        """Test that ``ValueError`` is being raised if ``self.trials`` is equal to
        ``self.tunable.SC``.
        """
        # setup
        self.instance.tunable.SC = 2
        self.instance.trials = np.array([[1], [2]])

        # run / assert
        with self.assertRaises(ValueError):
            self.instance._check_proposals(1)

    def test__check_proposals_trials_and_proposals_gt_sc(self):
        """Test that ``ValueError`` is being raised if ``proposals`` + ``len(self.trials)``
        is greater than ``self.tunable.SC``.
        """
        # setup
        self.instance.tunable.SC = 4
        self.instance.trials = np.array([[1], [2]])

        # run / assert
        with self.assertRaises(ValueError):
            self.instance._check_proposals(3)

    def test__check_proposals_not_raise(self):
        """Test that ``ValueError`` is not being raised."""
        # setup
        self.instance.tunable.SC = 4
        self.instance.trials = np.array([[1], [2]])

        # run / assert
        result = self.instance._check_proposals(1)
        assert result is None

    @patch('btb.tuning.tuners.base.BaseTuner._check_proposals')
    def test_propose_one_value_no_duplicates(self, mock__check_proposals):
        """Test that ``propose`` method calls it's child implemented method with
        ``allow_duplicates`` as ``False``.
        """
        # setup
        inverse_return = self.instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1]
        self.instance._propose = MagicMock(return_value=1)

        # run
        result = self.instance.propose(1)

        # assert
        self.instance._propose.assert_called_once_with(1, False)
        self.instance.tunable.inverse_transform.called_once_with(1)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == 1

    @patch('btb.tuning.tuners.base.BaseTuner._check_proposals')
    def test_propose_one_value_allow_duplicates(self, mock__check_proposals):
        """Test that ``propose`` method calls it's child implemented method with
        ``allow_duplicates`` as ``True``.
        """
        # setup
        inverse_return = self.instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1]
        self.instance._propose = MagicMock(return_value=1)

        # run
        result = self.instance.propose(1, allow_duplicates=True)

        # assert
        self.instance._propose.assert_called_once_with(1, True)
        self.instance.tunable.inverse_transform.called_once_with(1)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == 1

    @patch('btb.tuning.tuners.base.BaseTuner._check_proposals')
    def test_propose_many_values_no_duplicates(self, mock__check_proposals):
        """Test that ``propose`` method calls it's child implemented method with more than one
        proposals and ``allow_duplicates`` as ``False``.
        """

        # setup
        inverse_return = self.instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1, 2]
        self.instance._propose = MagicMock(return_value=2)

        # run
        result = self.instance.propose(2)

        # assert
        self.instance._propose.assert_called_once_with(2, False)
        self.instance.tunable.inverse_transform.called_once_with(2)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == [1, 2]

    @patch('btb.tuning.tuners.base.BaseTuner._check_proposals')
    def test_propose_many_values_allow_duplicates(self, mock__check_proposals):
        """Test that ``propose`` method calls it's child implemented method with more than one
        proposals and ``allow_duplicates`` as ``True``.
        """

        # setup
        inverse_return = self.instance.tunable.inverse_transform.return_value
        inverse_return.to_dict.return_value = [1, 2]
        self.instance._propose = MagicMock(return_value=2)

        # run
        result = self.instance.propose(2, allow_duplicates=True)

        # assert
        self.instance._propose.assert_called_once_with(2, True)
        self.instance.tunable.inverse_transform.called_once_with(2)
        inverse_return.to_dict.assert_called_once_with(orient='records')
        assert result == [1, 2]

    def test__sample_allow_duplicates(self):
        """Test the method ``_sample``when using duplicates."""
        # setup
        self.instance.tunable.sample.return_value = 1

        # run
        result = self.instance._sample(1, True)

        # assert
        self.instance.tunable.sample.assert_called_once_with(1)
        assert result == 1

    def test__sample_not_allow_duplicates(self):
        """Test that the method ``_sample`` returns ``np.ndarray`` when not using duplicates."""
        # setup
        self.instance.trials = np.array([[1], [2]])
        self.instance.tunable.sample.return_value = np.array([[3]])

        # run
        result = self.instance._sample(1, False)

        # assert
        self.instance.tunable.sample.assert_called_once_with(1)
        np.testing.assert_array_equal(result, np.array([[3]]))

    def test_sample_no_duplicates_more_than_one_loop(self):
        """Test that the method ``_sample`` returns ``np.ndarray`` when not using duplicates and
        perfroms more than one iteration.
        """
        # setup
        self.instance.trials = np.array([[1], [2]])

        side_effect = [np.array([[3]]), np.array([[1]]), np.array([[1]]), np.array([[4]])]
        self.instance.tunable.sample.side_effect = side_effect

        # run
        result = self.instance._sample(2, False)

        # assert
        assert self.instance.tunable.sample.call_args_list == [call(2), call(2), call(2), call(2)]
        np.testing.assert_array_equal(result, np.array([[3], [4]]))

    def test_record_list(self):
        """Test that the method record updates the ``trials``  and ``scores``."""

        # setup
        self.instance.tunable.transform.return_value = np.array([[1, 0]])

        # run
        self.instance.record([1], [0.1])

        # assert
        self.instance.tunable.transform.assert_called_once_with([1])

        np.testing.assert_array_equal(self.instance.trials, np.array([[1, 0]]))
        np.testing.assert_array_equal(self.instance.scores, np.array([0.1]))

    def test_record_scalar_values(self):
        """Test that the method record performs an update to ``trials`` and ``scores`` when called
        with a scalar value.
        """
        # setup
        self.instance.tunable.transform.return_value = np.array([[1, 0]])

        # run
        self.instance.record(1, 0.1)

        # assert
        self.instance.tunable.transform.assert_called_once_with(1)
        np.testing.assert_array_equal(self.instance.trials, np.array([[1, 0]]))
        np.testing.assert_array_equal(self.instance.scores, np.array([0.1]))

    def test_record_raise_error(self):
        """Test that the method record raises a ``ValueError`` when ``len(trials)`` is different
        to ``len(scores)``.
        """
        # setup
        self.instance.tunable.transform.return_value = np.array([[1, 0]])

        # run / assert
        with self.assertRaises(ValueError):
            self.instance.record(1, [1, 2])
