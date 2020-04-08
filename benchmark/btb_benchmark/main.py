# -*- coding: utf-8 -*-
import argparse
import logging
import random
import warnings
from datetime import datetime

import dask
import pandas as pd
import tabulate
from btb_benchmark.challenges import (
    MATH_CHALLENGES, Challenge, RandomForestChallenge, SGDChallenge, XGBoostChallenge)
from btb_benchmark.tuners import get_all_tuners

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


def _evaluate_tuners_on_challenges(tuners, challenges=None, iterations=1000):
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

        for name in tuners:
            tuning_function = all_tuners.get(name)

            if tuning_function:
                LOGGER.info('Loading tuning function: %s', name)
                selected_tuning_functions[name] = tuning_function

            else:
                LOGGER.info('Could not load tuning function: %s', name)

        if not selected_tuning_functions:
            raise ValueError('No tunable function was loaded.')

        return selected_tuning_functions


def _get_all_challenge_names(types=None):
    all_challenge_names = []
    types = types or ALL_TYPES
    if 'math' in types:
        all_challenge_names += list(MATH_CHALLENGES.keys())

    # Using elif as the datasets are the same
    if 'sgd' in types:
        all_challenge_names += SGDChallenge.get_available_dataset_names()
    elif 'xgboost' in types:
        all_challenge_names += XGBoostChallenge.get_available_dataset_names()
    elif 'random_forest' in types:
        all_challenge_names += RandomForestChallenge.get_available_dataset_names()

    return all_challenge_names


def _get_challenges(challenges=None, types=None, samples=None):
    challenges = challenges or _get_all_challenge_names(types)
    selected = []
    unknown = []

    if samples:
        if samples > len(challenges):
            raise ValueError('Samples can not be greater than {}'.format(len(challenges)))

        challenges = random.sample(challenges, samples)

    for challenge_name in challenges:
        known = False
        for challenge_type in types or ALL_TYPES:
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


def _sanitize_param(param):
    """If the param its a `str` return it as a `list`, else return the param even if it's None."""
    if isinstance(param, str):
        return [param]

    return param


def run_benchmark(types='xgboost', tuners='BTB.GPTuner', challenges=None,
                  samples=None, iterations=10, output_path=None):
    """Run Benchmark function.

    Args:
        types (str or list):
            Type or list of types for challenges to be benchmarked. Defaults to ``xgboost``.

        tuners (str or list):
            Tuner name or list of tuner names to be benchmarked. Defaults to ``BTB.GPTuner``.

        challenges (str or list):
            Challenge name or list of challenge names to be benchmarked.

        samples (int):
            Amount of ``challenges`` to be benchmarked randomly. Defaults to ``None``.

        iterations (int):
            Number of tuning iterations to perform.

        output_path (str):
            If an ``output_path`` is given, the final results will be saved in that location.

    Returns:
        pandas.DataFrame

    """

    types = _sanitize_param(types)
    challenges = _sanitize_param(challenges)

    if not isinstance(tuners, dict):
        tuners = _sanitize_param(tuners)
        tuners = _get_tuners(tuners)

    challenges = _get_challenges(challenges=challenges, types=types, samples=samples)
    results = _evaluate_tuners_on_challenges(tuners, challenges, iterations)

    if output_path:
        LOGGER.info('Saving benchmark report to %s', output_path)
        results.to_csv(output_path)

    return results


def _get_parser():
    parser = argparse.ArgumentParser(description='BTB Benchmark Command Line Interface')

    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')
    parser.add_argument('-o', '--output-path', type=str, required=False,
                        help='Path to the CSV file where the report will be dumped')
    parser.add_argument('-s', '--samples', type=int,
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

    # run
    results = run_benchmark(
        args.types,
        args.tuners,
        args.challenges,
        args.samples,
        args.iterations,
        args.output_path,
    )

    print(tabulate.tabulate(
        results,
        tablefmt='github',
        headers=results.columns
    ))


if __name__ == '__main__':
    main()
