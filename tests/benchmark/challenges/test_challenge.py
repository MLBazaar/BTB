# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch

from btb.benchmark.challenges.challenge import MLChallenge


class TestMLChallenge(TestCase):

    @patch('btb.benchmark.challenges.challenge.make_scorer')
    @patch('btb.benchmark.challenges.challenge.StratifiedKFold')
    @patch('btb.benchmark.challenges.challenge.MLChallenge.load_data')
    def test___init__stratified(self, mock_load_data, mock_strfkfold, mock_make_scorer):

        # setup
        mock_load_data.return_value = (1, 2)
        mock_make_scorer.return_value = 'test_scorer'
        mock_strfkfold.return_value = 'cv'

        # run
        instance = MLChallenge(
            model='test',
            dataset='any',
            target_column='test_column',
            encode=False,
            tunable_hyperparameters='test_hp',
            metric='f1_score',
            model_defaults='any',
            make_binary=True,
            stratified=True
        )

        # assert
        assert instance.model == 'test'
        assert instance.dataset == 'any'
        assert instance.model == 'test'
        assert instance.dataset == 'any'
        assert instance.target_column == 'test_column'
        assert not instance.encode
        assert instance.tunable_hyperparameters == 'test_hp'
        assert instance.scorer == 'test_scorer'
        assert instance.model_defaults == 'any'
        assert instance.make_binary
        assert instance.cv == 'cv'

        mock_strfkfold.assert_called_once_with(shuffle=True, n_splits=5, random_state=42)

    @patch('btb.benchmark.challenges.challenge.make_scorer')
    @patch('btb.benchmark.challenges.challenge.KFold')
    @patch('btb.benchmark.challenges.challenge.MLChallenge.load_data')
    def test___init__kfold(self, mock_load_data, mock_kfold, mock_make_scorer):

        # setup
        mock_load_data.return_value = (1, 2)
        mock_make_scorer.return_value = 'test_scorer'
        mock_kfold.return_value = 'kfold'

        # run
        instance = MLChallenge(
            model='test',
            dataset='any',
            target_column='test_column',
            encode=False,
            tunable_hyperparameters='test_hp',
            metric='f1_score',
            model_defaults='any',
            make_binary=True,
            stratified=False
        )

        # assert
        assert instance.model == 'test'
        assert instance.dataset == 'any'
        assert instance.model == 'test'
        assert instance.dataset == 'any'
        assert instance.target_column == 'test_column'
        assert not instance.encode
        assert instance.tunable_hyperparameters == 'test_hp'
        assert instance.scorer == 'test_scorer'
        assert instance.model_defaults == 'any'
        assert instance.make_binary
        assert instance.cv == 'kfold'

        mock_kfold.assert_called_once_with(shuffle=True, n_splits=5, random_state=42)

    @patch('btb.benchmark.challenges.challenge.make_scorer')
    @patch('btb.benchmark.challenges.challenge.StratifiedKFold')
    @patch('btb.benchmark.challenges.challenge.OneHotEncoder')
    def test___init__encode(self, mock_ohe, mock_strfkfold, mock_make_scorer):

        # setup
        instance = MagicMock()
        instance.load_data.return_value = ('X', 'y')
        instance.encode = True
        instance.DATASET = 'dataset'

        # run
        MLChallenge.__init__(instance, metric='test', encode=True)

        # assert
        mock_ohe.return_value.fit_transform.assert_called_once_with('X')

    def test_get_tunable_hyperparameters(self):
        # setup
        instance = MagicMock()
        instance.tunable_hyperparameters = {'test': 'hyperparam'}

        # run
        result = MLChallenge.get_tunable_hyperparameters(instance)

        # assert
        assert result == {'test': 'hyperparam'}

    @patch('btb.benchmark.challenges.challenge.cross_val_score')
    def test_evaluate(self, mock_crossval):
        # setup
        mock_crossval.return_value.mean.return_value = 1
        instance = MagicMock()
        instance.scorer = 'scoring'
        instance.cv = 'cv'

        # run
        result = MLChallenge.evaluate(instance)

        # assert
        assert result == 1

        mock_crossval(instance.model.return_value, 'X', 'y', cv='cv', scoring='scoring')
