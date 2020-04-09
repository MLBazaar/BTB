# -*- coding: utf-8 -*-
import argparse
import logging
import os
import random
import warnings
from datetime import datetime

import dask
import pandas as pd

import btb
from btb.tuning.tuners.base import BaseTuner
from btb_benchmark.challenges import (
    MATH_CHALLENGES, Challenge, RandomForestChallenge, SGDChallenge, XGBoostChallenge)
from btb_benchmark.tuners import get_all_tuners
from btb_benchmark.tuners.btb import make_btb_tuning_function

LOGGER = logging.getLogger(__name__)
ALL_TYPES = ['math', 'xgboost']

CHALLENGE_GETTER = {
    'math': MATH_CHALLENGES.get,
    'random_forest': RandomForestChallenge,
    'sgd': SGDChallenge,
    'xgboost': XGBoostChallenge,
}

warnings.filterwarnings("ignore")


@dask.delayed
def _evaluate_tuner_on_challenge(name, tuner, challenge, iterations):
    tunable_hyperparameters = challenge.get_tunable_hyperparameters()
    LOGGER.info('Evaluating tuner %s on challenge %s for %s iterations',
                name, challenge, iterations)
    try:
        start = datetime.utcnow()
        score = tuner(challenge.evaluate, tunable_hyperparameters, iterations)
        result = {
            'challenge': str(challenge),
            'tuner': name,
            'score': score,
            'iterations': iterations,
            'elapsed': datetime.utcnow() - start
        }

    except Exception as ex:
        LOGGER.warn(
            'Could not score tuner %s with challenge %s, error: %s', name, challenge, ex)
        result = {
            'challenge': str(challenge),
            'tuner': name,
            'score': None,
            'elapsed': datetime.utcnow() - start
        }

    return result


def _evaluate_tuner_on_challenges(name, tuner, challenges, iterations):
    tuner_results = []
    for challenge in challenges:
        try:
            if not isinstance(challenge, Challenge) and issubclass(challenge, Challenge):
                challenge = challenge()

            result = _evaluate_tuner_on_challenge(name, tuner, challenge, iterations)

        except Exception as ex:
            LOGGER.warn(
                'Could not score tuner %s with challenge %s, error: %s', name, challenge, ex)

        tuner_results.append(result)

    return tuner_results


def benchmark(tuners, challenges=None, iterations=1000):
    """Benchmark function.

    This benchmark function iterates over a collection of ``challenges`` and executes a
    ``tuner_function`` for each one of the ``challenges`` for a given amount of iterations.

    Args:
        tuners (callable, list, tuple or dict):
            Python callable function, list of callable functions, tuple with callable functions or
            dictionary with the ``name`` of the function as ``key`` and the callable function that
            returns the best score for a given ``scorer``. This function must have three arguments:

                * scorer (function):
                    A function that performs scoring over params.
                * tunable (btb.tuning.Tunable):
                    A ``Tunable`` instance used to instantiate a tuner.
                * iterations (int):
                    Number of tuning iterations to perform.

        challenges (single challenge or list):
            A single ``challenge`` or a list of ``chalenges``. This challenges must inherit
            from ``btb.challenges.challenge.Challenge``.
        iterations (int):
            Amount of iterations to perform for the ``tuner_function``.

    Returns:
        pandas.DataFrame:
            A ``pandas.DataFrame`` with the obtained scores for the given challenges is being
            returned.
    """
    if challenges is None:
        challenges = list(MATH_CHALLENGES.values())
    elif not isinstance(challenges, list):
        challenges = [challenges]

    if callable(tuners):
        tuners = {tuners.__name__: tuners}
    elif isinstance(tuners, (list, tuple)):
        tuners = {tuner.__name__: tuner for tuner in tuners}
    elif not isinstance(tuners, dict):
        raise TypeError('tuners can only be a callable, list or dict.')

    results = []

    for name, tuner in tuners.items():
        LOGGER.info('Evaluating tuner %s', name)
        result = _evaluate_tuner_on_challenges(name, tuner, challenges, iterations)
        results.extend(result)

    results = [result for result in dask.compute(*results)]

    df = pd.DataFrame.from_records(results)
    df = df.pivot(index='challenge', columns='tuner', values='score')
    del df.columns.name
    del df.index.name

    return df


def _get_tuners(tuners=None):
    all_tuners = get_all_tuners()
    if tuners is None:
        LOGGER.info('Using all tuning functions.')
        return all_tuners
    else:
        selected_tuning_functions = {}
        tuners = _as_list(tuners)

        for tuner in tuners:
            if not isinstance(tuner, str) and issubclass(tuner, BaseTuner):
                selected_tuning_functions[tuner.__name__] = make_btb_tuning_function(tuner)
            else:
                tuning_function = all_tuners.get(tuner)
                if tuning_function:
                    LOGGER.info('Loading tuning function: %s', tuner)
                    selected_tuning_functions[tuner] = tuning_function
                else:
                    LOGGER.info('Could not load tuning function: %s', tuner)

        if not selected_tuning_functions:
            raise ValueError('No tunable function was loaded.')

        return selected_tuning_functions


def _get_all_challenge_names(types=None):
    all_challenge_names = []
    types = types or ALL_TYPES

    if 'math' in types:
        all_challenge_names += list(MATH_CHALLENGES.keys())

    if any(name in types for name in ('sdg', 'xgboost', 'random_forest')):
        all_challenge_names += SGDChallenge.get_available_dataset_names()

    return all_challenge_names


def _get_challenges(challenges=None, types=None, sample=None):
    types = _as_list(types) or ALL_TYPES
    challenges = _as_list(challenges) or _get_all_challenge_names(types)
    selected = []
    unknown = []

    if sample:
        if sample > len(challenges):
            raise ValueError('Sample can not be greater than {}'.format(len(challenges)))

        challenges = random.sample(challenges, sample)

    for challenge_name in challenges:
        known = False
        if not isinstance(challenge_name, str) and issubclass(challenge_name, Challenge):
            selected.append(challenge_name)
        else:
            for challenge_type in types:
                try:
                    challenge = CHALLENGE_GETTER[challenge_type](challenge_name)
                    if challenge:
                        known = True
                        selected.append(challenge)
                except Exception:
                    pass

            if not known:
                unknown.append(challenge_name)

    if unknown:
        raise ValueError('Challenges {} not of type {}'.format(unknown, types))

    if not selected:
        raise ValueError('No challenges selected!')

    return selected


def _as_list(param):
    """If the param its a ``str`` return it as a ``list``, else return the param."""
    if not isinstance(param, (list, tuple)) and param:
        return [param]

    return param


def run_benchmark(types=None, tuners=None, challenges=None,
                  sample=None, iterations=100, output_path=None):
    """Run Benchmark.

    The ``run_benchmark`` function provides a user friendly interface to launch a ``benchmark``
    process that evaluates the performance of a ``tuner`` or a list of ``tuners`` against a
    ``challenge`` or a list of ``challenges`` for a given amount of iterations.

    This function also allows to export the results in a given ``output_path`` where it will
    be saved as a ``csv`` file.

    Args:
        types (str or list):
            Type or list of types for challenges to be benchmarked, if ``None`` all available
            types will be used.
        tuners (str, btb.tuners.base.BaseTuner or list):
            Tuner name, ``btb.tuners.base.BaseTuner`` subclass or a list with the previously
            described objects. If ``None`` all available ``tuners`` implemented in
            ``btb_benchmark`` will be used.
        challenges (str, btb_benchmark.challenge.Challenge or list):
            Challenge name, ``btb_benchmark.challenge.Challenge`` subclass or a list with the
            previously described objects. If ``None`` will use ``types`` to determine which
            challenges to use.
        sample (int):
            Amount of ``challenges`` to be benchmarked randomly. Defaults to ``None``. If
            ``types`` is given, will randomly sample from those challenges.
        iterations (int):
            Number of tuning iterations to perform per challenge and tuner.
        output_path (str):
            If an ``output_path`` is given, the final results will be saved in that location.

    Returns:
        pandas.DataFrame or None:
            If ``output_path`` is ``None`` it will return a ``pandas.DataFrame`` object,
            else it will dump the results in the specified ``output_path``.
    """
    tuners = _get_tuners(tuners)
    challenges = _get_challenges(challenges=challenges, types=types, sample=sample)
    results = benchmark(tuners, challenges, iterations)

    if output_path:
        LOGGER.info('Saving benchmark report to %s', output_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, '{}.csv'.format(btb.__version__))

        results.to_csv(output_path)

    else:
        return results


def _get_parser():
    parser = argparse.ArgumentParser(description='BTB Benchmark Command Line Interface')

    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')
    parser.add_argument('-o', '--output-path', type=str, required=False,
                        help='Path to the CSV file where the report will be dumped')
    parser.add_argument('-s', '--sample', type=int,
                        help='Limit the test to a sample of datasets for the given size.')
    parser.add_argument('-i', '--iterations', type=int, default=100,
                        help='Number of iterations to perform per challenge with each candidate.')
    parser.add_argument('--challenges', nargs='+', help='Name of the challenge/s to be processed.')
    parser.add_argument('--tuners', nargs='+', help='Name of the tunables to be used.')
    parser.add_argument('--types', nargs='+', help='Name of the tunables to be used.',
                        choices=['math', 'sgd', 'random_forest', 'xgboost'])

    return parser


def main():
    # Parse args
    parser = _get_parser()
    args = parser.parse_args()

    # Logger setup
    log_level = (3 - args.verbose) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(level=log_level, format=fmt)
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("hyperopt").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    if args.output_path is None:
        args.output_path = '{}.csv'.format(btb.__version__)

    # run
    run_benchmark(
        args.types,
        args.tuners,
        args.challenges,
        args.sample,
        args.iterations,
        args.output_path,
    )


if __name__ == '__main__':
    main()
