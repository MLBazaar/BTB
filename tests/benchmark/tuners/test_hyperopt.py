# -*- coding: utf-8 -*-

from unittest.mock import call, patch

import pytest

from btb.benchmark.tuners.hyperopt import make_minimize_function, search_space_from_dict


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
    mock_hyperopt_hp.choice.side_effect = ('choice_hyperparam', 'bool_hyperparam')

    dict_hyperparams_cat = {
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
        }
    }

    dict_hyperparams_bool = {
        'bool': {
            'type': 'bool',
        }
    }

    # run
    res_cat = search_space_from_dict(dict_hyperparams_cat)
    res_bool = search_space_from_dict(dict_hyperparams_bool)

    # assert
    expected_res_cat = {
        'int': 'int_hyperparam',
        'float': 'float_hyperparam',
        'cat': 'choice_hyperparam',
    }
    expected_res_bool = {
        'bool': 'bool_hyperparam',
    }

    expected_call_choice = [call('cat', ['a', 'b']), call('bool', [True, False])]

    assert res_cat == expected_res_cat
    assert res_bool == expected_res_bool

    mock_hyperopt_hp.uniformint.assert_called_once_with('int', 1, 2)
    mock_hyperopt_hp.uniform.assert_called_once_with('float', 0, 1)
    assert mock_hyperopt_hp.choice.call_args_list == expected_call_choice


def test_make_minimize_function():
    # setup
    def scoring_function(a):
        return a

    # run
    res_func = make_minimize_function(scoring_function)
    result = res_func({'a': 1})

    # assert
    result == -1
