# -*- coding: utf-8 -*-
import argparse
import logging
import random
import warnings
from datetime import datetime

import dask
import pandas as pd

from btb.tuning.tuners.base import BaseTuner
from btb_benchmark.challenges import (
    MATH_CHALLENGES, Challenge, RandomForestChallenge, SGDChallenge, XGBoostChallenge)
from btb_benchmark.results import load_results, write_results
from btb_benchmark.tuners import get_all_tuners
from btb_benchmark.tuners.btb import make_btb_tuning_function

LOGGER = logging.getLogger(__name__)
ALL_TYPES = ['math', 'xgboost']

warnings.filterwarnings("ignore")


def get_math_challenge_instance(name):
    return MATH_CHALLENGES.get(name)()


CHALLENGE_GETTER = {
    'math': get_math_challenge_instance,
    'random_forest': RandomForestChallenge,
    'sgd': SGDChallenge,
    'xgboost': XGBoostChallenge,
}


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
            result = _evaluate_tuner_on_challenge(name, tuner, challenge, iterations)
        except Exception as ex:
            LOGGER.warn(
                'Could not score tuner %s with challenge %s, error: %s', name, challenge, ex)

        tuner_results.append(result)

    return tuner_results


def benchmark(tuners, challenges, iterations):
    """Score ``tuners`` against a list of ``challenges`` for the given amount of iterations.

    This function scores a collection of ``tuners`` against a collection of ``challenges``
    performing tuning iterations in order to obtain a better score. At the end, the best score
    for each tuner / challenge is being returned. This data is returned as a ``pandas.DataFrame``.

    Args:
        tuners (dict):
            Python dictionary with the ``name`` of the function as ``key`` and the callable
            function that returns the best score for a given ``scorer``.
            This function must have three arguments:

                * scorer (function):
                    A function that performs scoring over params.
                * tunable (btb.tuning.Tunable):
                    A ``Tunable`` instance used to instantiate a tuner.
                * iterations (int):
                    Number of tuning iterations to perform.

        challenges (list):
            A list of ``chalenges``. This challenges must inherit from
            ``btb.challenges.challenge.Challenge``.
        iterations (int):
            Amount of tuning iterations to perform for each tuner and each challenge.

    Returns:
        pandas.DataFrame:
            A ``pandas.DataFrame`` with the obtained scores for the given challenges is being
            returned.
    """
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


def _as_list(param):
    """Make sure that param is either ``None`` or a ``list``."""
    if param is None or isinstance(param, (list, tuple)):
        return param

    return [param]


def _get_tuners_dict(tuners=None):
    all_tuners = get_all_tuners()
    if tuners is None:
        LOGGER.info('Using all tuning functions.')
        return all_tuners
    else:
        selected_tuning_functions = {}
        for tuner in _as_list(tuners):
            if isinstance(tuner, type) and issubclass(tuner, BaseTuner):
                selected_tuning_functions[tuner.__name__] = make_btb_tuning_function(tuner)
            elif callable(tuner):
                selected_tuning_functions[tuner.__name__] = tuner
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


def _get_all_challenge_names(challenge_types=None):
    all_challenge_names = []
    if 'math' in challenge_types:
        all_challenge_names += list(MATH_CHALLENGES.keys())
    if any(name in challenge_types for name in ('sdg', 'xgboost', 'random_forest')):
        all_challenge_names += SGDChallenge.get_available_dataset_names()

    return all_challenge_names


def _get_challenges_list(challenges=None, challenge_types=None, sample=None):
    challenge_types = _as_list(challenge_types) or ALL_TYPES
    challenges = _as_list(challenges) or _get_all_challenge_names(challenge_types)
    selected = []
    unknown = []

    if sample:
        if sample > len(challenges):
            raise ValueError('Sample can not be greater than {}'.format(len(challenges)))

        challenges = random.sample(challenges, sample)

    for challenge in challenges:
        known = False
        if isinstance(challenge, Challenge):
            selected.append(challenge)
        else:
            for challenge_type in challenge_types:
                try:
                    challenge_instance = CHALLENGE_GETTER[challenge_type](challenge)
                    if challenge_instance:
                        known = True
                        selected.append(challenge_instance)
                except Exception:
                    pass

            if not known:
                unknown.append(challenge)

    if unknown:
        raise ValueError('Challenges {} not of type {}'.format(unknown, challenge_types))

    if not selected:
        raise ValueError('No challenges selected!')

    return selected


def run_benchmark(tuners=None, challenge_types=None, challenges=None,
                  sample=None, iterations=100, output_path=None):
    """Execute the benchmark function and optionally store the result as a ``CSV``.

    This function provides a user-friendly interface to interact with the ``benchmark``
    function. It allows the user to specify an ``output_path`` where the results can be
    stored. If this path is not provided, a ``pandas.DataFrame`` will be returned.

    Args:
        tuners (str, btb.tuning.tuners.base.BaseTuner or list):
            Tuner name, ``btb.tuning.tuners.base.BaseTuner`` subclass or a list with the previously
            described objects. If ``None`` all available ``tuners`` implemented in
            ``btb_benchmark`` will be used.
        challenge_types (str or list):
            Type or list of types for challenges to be benchmarked, if ``None`` all available
            types will be used.
        challenges (str, btb_benchmark.challenge.Challenge or list):
            Challenge name, ``btb_benchmark.challenge.Challenge`` instance or a list with the
            previously described objects. If ``None`` will use ``challenge_types`` to determine
            which challenges to use.
        sample (int):
            Run only on a subset of the available datasets of the given size.
        iterations (int):
            Number of tuning iterations to perform per challenge and tuner.
        output_path (str):
            If an ``output_path`` is given, the final results will be saved in that location.

    Returns:
        pandas.DataFrame or None:
            If ``output_path`` is ``None`` it will return a ``pandas.DataFrame`` object,
            else it will dump the results in the specified ``output_path``.
    """
    tuners = _get_tuners_dict(tuners)
    challenges = _get_challenges_list(
        challenges=challenges,
        challenge_types=challenge_types,
        sample=sample
    )
    results = benchmark(tuners, challenges, iterations)

    if output_path:
        LOGGER.info('Saving benchmark report to %s', output_path)
        results.to_csv(output_path)

    else:
        return results


def summarize_results(input_paths, output_path):
    """Load multiple benchmark results CSV files and compile a summary.

    The result is an Excel file with one tab for each results CSV file
    and an additional Number of Wins tab with a summary of the number
    of challenges in which each Tuner got the best score.

    Args:
        inputs_paths (list[str]):
            List of paths to CSV files where the benchmarks results are stored.
            These files must have one column per Tuner and one row per Challenge.
        output_path (str):
            Path, including the filename, where the Excel file will be created.
    """
    results = load_results(input_paths)
    write_results(results, output_path)


def _run(args):
    # Logger setup
    log_level = (3 - args.verbose) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(level=log_level, format=fmt)
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("hyperopt").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # run
    run_benchmark(
        args.tuners,
        args.challenge_types,
        args.challenges,
        args.sample,
        args.iterations,
        args.output_path,
    )


def _summary(args):
    summarize_results(args.input, args.output)


def _get_parser():
    parser = argparse.ArgumentParser(description='BTB Benchmark Command Line Interface')
    parser.set_defaults(action=None)
    action = parser.add_subparsers(title='action')
    action.required = True

    # Run action
    run = action.add_parser('run', help='Run the BTB Benchmark')
    run.set_defaults(action=_run)
    run.set_defaults(user=None)

    run.add_argument('-v', '--verbose', action='count', default=0,
                     help='Be verbose. Use -vv for increased verbosity.')
    run.add_argument('-o', '--output-path', type=str, required=False,
                     help='Path to the CSV file where the report will be dumped')
    run.add_argument('-s', '--sample', type=int,
                     help='Run only on a subset of the available datasets of the given size.')
    run.add_argument('-i', '--iterations', type=int, default=100,
                     help='Number of iterations to perform per challenge with each candidate.')
    run.add_argument('-c', '--challenges', nargs='+',
                     help='Challenge/s to be used. Accepts multiple names.')
    run.add_argument('-t', '--tuners', nargs='+',
                     help='Tuner/s to be benchmarked. Accepts multiple names.')
    run.add_argument('-C', '--challenge-types', nargs='+',
                     choices=['math', 'sgd', 'random_forest', 'xgboost'],
                     help='Type of challenge/s to use. Accepts multiple names.')

    # Summarize action
    summary = action.add_parser('summary', help='Summarize the BTB Benchmark results')
    summary.set_defaults(action=_summary)
    summary.add_argument('input', nargs='+', help='Input path with results.')
    summary.add_argument('output', help='Output file.')

    return parser


def main():
    # Parse args
    parser = _get_parser()
    args = parser.parse_args()

    args.action(args)


if __name__ == '__main__':
    main()
