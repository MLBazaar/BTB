# -*- coding: utf-8 -*-

from unittest.mock import call, patch

import pytest
from hyperopt import tpe

from btb.benchmark.tuners.hyperopt import (
    hyperopt_tuning_function, sanitize_scoring_function, search_space_from_dict)


def test_search_space_from_dict_not_dict():
    """Test case for method."""
    # setup
    hyperparams = list()

    # run
    with pytest.raises(TypeError):
        search_space_from_dict(hyperparams)


@patch('btb.benchmark.tuners.hyperopt.hp')
def test_search_space_from_dict(mock_hyperopt_hp):
    # setup
    mock_hyperopt_hp.uniform.return_value = 'float_hyperparam'
    mock_hyperopt_hp.uniformint.return_value = 'int_hyperparam'
    mock_hyperopt_hp.choice.side_effect = ['choice_hyperparam', 'bool_hyperparam']

    dict_hyperparams = {
        'int': {
            'type': 'int',
            'range': [1, 2],
        },
        'float': {
            'type': 'float',
            'range': [0, 1],
        },
        'cat': {
            'type': 'str',
            'range': ['a', 'b'],
        },
        'bool': {
            'type': 'bool',
        },
    }

    # run
    res = search_space_from_dict(dict_hyperparams)

    # assert
    expected_res = {
        'int': 'int_hyperparam',
        'float': 'float_hyperparam',
        'cat': 'choice_hyperparam',
        'bool': 'bool_hyperparam',
    }

    expected_call_choice = [call('cat', ['a', 'b']), call('bool', [True, False])]
    assert res == expected_res

    mock_hyperopt_hp.uniformint.assert_called_once_with('int', 1, 2)
    mock_hyperopt_hp.uniform.assert_called_once_with('float', 0, 1)
    assert mock_hyperopt_hp.choice.call_args_list == expected_call_choice


def test_sanitize_scoring_function():
    # setup
    def scoring_function(a):
        return a

    # run
    res_func = sanitize_scoring_function(scoring_function)
    result = res_func({'a': 1})

    # assert
    result == -1


@patch('btb.benchmark.tuners.hyperopt.fmin')
@patch('btb.benchmark.tuners.hyperopt.Trials')
@patch('btb.benchmark.tuners.hyperopt.search_space_from_dict')
@patch('btb.benchmark.tuners.hyperopt.sanitize_scoring_function')
def test_hyperopt_tuning_function(mock_sanitize, mock_ss_from_dict, mock_trials, mock_fmin):
    # setup
    mock_sanitize.return_value = 'sanitized'
    mock_ss_from_dict.return_value = 'search_space'
    mock_trials.return_value.best_trial = {'result': {'loss': -1}}

    # run
    result = hyperopt_tuning_function('test_scoring_func', 'tunable_hps', 10)

    # assert
    assert result == 1

    mock_sanitize.assert_called_once_with('test_scoring_func')
    mock_ss_from_dict.assert_called_once_with('tunable_hps')
    mock_trials.assert_called_once_with()

    mock_fmin.assert_called_once_with(
        'sanitized',
        'search_space',
        algo=tpe.suggest,
        max_evals=10,
        trials=mock_trials.return_value
    )
