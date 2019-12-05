# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch

from btb.benchmark.challenges.census import DEFAULT_CV, DEFAULT_SCORER, CensusRF


class TestCensusRF(TestCase):

    @patch('btb.benchmark.challenges.census.OneHotEncoder')
    @patch('btb.benchmark.challenges.census.CensusRF.load_data')
    def test___init__default(self, mock_load_data, mock_ohe):
        # setup
        mock_load_data.return_value = (1, 2)
        mock_ohe.return_value.fit_transform.return_value = 'X'

        # run
        census = CensusRF()

        # assert
        assert census.X == 'X'
        assert census.y == 2
        assert census.scorer == DEFAULT_SCORER
        assert census.cv == DEFAULT_CV

        mock_load_data.assert_called_once_with()
        mock_ohe.assert_called_once_with()
        mock_ohe.return_value.fit_transform.assert_called_once_with(1)

    @patch('btb.benchmark.challenges.census.OneHotEncoder')
    @patch('btb.benchmark.challenges.census.CensusRF.load_data')
    def test___init__custom(self, mock_load_data, mock_ohe):
        # run
        mock_load_data.return_value = (1, 2)
        mock_ohe.return_value.fit_transform.return_value = 'X'
        census = CensusRF(cv='cv_test', scorer='scorer_test')

        # assert
        assert census.X == 'X'
        assert census.y == 2
        assert census.scorer == 'scorer_test'
        assert census.cv == 'cv_test'

        mock_load_data.assert_called_once_with()
        mock_ohe.assert_called_once_with()
        mock_ohe.return_value.fit_transform.assert_called_once_with(1)

    def test_get_tunable_hyperparameters(self):
        # run
        result = CensusRF().get_tunable_hyperparameters()

        # assert

        expected_result = {
            "n_estimators": {
                "type": "int",
                "default": 10,
                "range": [
                    1,
                    500
                ]
            },
            "criterion": {
                "type": "str",
                "default": "gini",
                "values": [
                    "entropy",
                    "gini"
                ]
            },
            "max_features": {
                "type": "str",
                "default": None,
                "values": [
                    None,
                    "auto",
                    "log2",
                    "sqrt"
                ]
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "range": [
                    2,
                    100
                ]
            },
            "min_samples_leaf": {
                "type": "int",
                "default": 1,
                "range": [
                    1,
                    100
                ]
            },
            "min_weight_fraction_leaf": {
                "type": "float",
                "default": 0.0,
                "range": [
                    0.0,
                    0.5
                ]
            },
            "min_impurity_decrease": {
                "type": "float",
                "default": 0.0,
                "range": [
                    0.0,
                    1000.0
                ]
            },
        }

        assert result == expected_result

    @patch('btb.benchmark.challenges.census.cross_val_score')
    @patch('btb.benchmark.challenges.census.OneHotEncoder')
    @patch('btb.benchmark.challenges.census.CensusRF.load_data')
    def test_evaluate(self, mock_load_data, mock_ohe, mock_cross_val):
        # setup
        mock_load_data.return_value = (1, 2)
        mock_cross_val.return_value.mean.return_value = 1

        census = CensusRF(cv='cv_test', scorer='scorer_test')
        census.MODEL_CLASS = MagicMock()

        # run
        result = census.evaluate()

        # assert
        assert result == 1
        mock_cross_val.assert_called_once_with(
            census.MODEL_CLASS.return_value,
            mock_ohe.return_value.fit_transform.return_value,
            2,
            cv='cv_test'
        )

        census.MODEL_CLASS.assert_called_once_with(random_state=0)

    @patch('btb.benchmark.challenges.census.cross_val_score')
    @patch('btb.benchmark.challenges.census.OneHotEncoder')
    @patch('btb.benchmark.challenges.census.CensusRF.load_data')
    def test_evaluate_hyperparams(self, mock_load_data, mock_ohe, mock_cross_val):
        # setup
        mock_load_data.return_value = (1, 2)
        mock_cross_val.return_value.mean.return_value = 1

        census = CensusRF(cv='cv_test', scorer='scorer_test')
        census.MODEL_CLASS = MagicMock()

        # run
        hyperparams = {'test': 'hyperparam'}
        result = census.evaluate(**hyperparams)

        # assert
        assert result == 1
        mock_cross_val.assert_called_once_with(
            census.MODEL_CLASS.return_value,
            mock_ohe.return_value.fit_transform.return_value,
            2,
            cv='cv_test'
        )

        census.MODEL_CLASS.assert_called_once_with(random_state=0, test='hyperparam')
