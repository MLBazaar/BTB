import argparse
import logging
import os
import random
import warnings

import tabulate

import btb
from btb.benchmark import benchmark
from btb.benchmark.challenges import (
    MATH_CHALLENGES, RandomForestChallenge, SGDChallenge, XGBoostChallenge)
from btb.benchmark.tuners import get_all_tuning_functions

LOGGER = logging.getLogger(__name__)
ALL_TYPES = ['math', 'xgboost']

warnings.filterwarnings("ignore")


def _get_candidates(args):
    all_tuning_functions = get_all_tuning_functions()

    if args.tuners is None:
        LOGGER.info('Using all tuning functions.')

        return all_tuning_functions

    else:
        selected_tuning_functions = {}

        for name in args.tuners:
            tuning_function = all_tuning_functions.get(name)

            if tuning_function:
                LOGGER.info('Loading tuning function: %s', name)
                selected_tuning_functions[name] = tuning_function

            else:
                LOGGER.info('Could not load tuning function: %s', name)

        if not selected_tuning_functions:
            raise ValueError('No tunable function was loaded.')

        return selected_tuning_functions


CHALLENGE_GETTER = {
    'math': MATH_CHALLENGES.get,
    'random_forest': RandomForestChallenge,
    'sgd': SGDChallenge,
    'xgboost': XGBoostChallenge,
}


def _get_all_challenges_names(args):
    all_challenge_names = []
    types = args.type or ALL_TYPES
    if 'math' in types:
        all_challenge_names += list(MATH_CHALLENGES.keys())

    # Using elif as the datasets are the same
    if 'sgd' in types:
        all_challenge_names += SGDChallenge.get_available_dataset_names()
    elif 'xgboost' in types:
        all_challenge_names += SGDChallenge.get_available_dataset_names()
    elif 'random_forest' in types:
        all_challenge_names += SGDChallenge.get_available_dataset_names()

    return all_challenge_names


def _get_challenges(args):
    challenges = args.challenges or _get_all_challenges_names(args)
    selected = []
    unknown = []

    if args.sample:
        if args.sample > len(challenges):
            raise ValueError('Sample can not be greater than {}'.format(len(challenges)))

        challenges = random.sample(challenges, args.sample)

    for challenge_name in challenges:
        known = False
        for challenge_type in args.type or ALL_TYPES:
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
        raise ValueError('Challenges {} not of type {}'.format(unknown, args.type))

    if not selected:
        raise ValueError('No challenges selected!')

    return selected


def perform_benchmark(args):
    candidates = _get_candidates(args)
    challenges = list(_get_challenges(args))
    results = benchmark(candidates, challenges, args.iterations)

    if args.report is None:
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        args.report = os.path.join(base_dir, '{}.csv'.format(btb.__version__))

    LOGGER.info('Saving benchmark report to %s', args.report)

    print(tabulate.tabulate(
        results,
        tablefmt='github',
        headers=results.columns
    ))

    results.to_csv(args.report)


def _get_parser():
    parser = argparse.ArgumentParser(description='BTB Benchmark Command Line Interface')

    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')
    parser.add_argument('-r', '--report', type=str, required=False,
                        help='Path to the CSV file where the report will be dumped')
    parser.add_argument('-s', '--sample', type=int,
                        help='Limit the test to a sample of datasets for the given size.')
    parser.add_argument('-i', '--iterations', type=int, default=100,
                        help='Number of iterations to perform per challenge with each candidate.')
    parser.add_argument('--challenges', nargs='+', help='Name of the challenge/s to be processed.')
    parser.add_argument('--tuners', nargs='+', help='Name of the tunables to be used.')
    parser.add_argument('--type', nargs='+', help='Name of the tunables to be used.',
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
    perform_benchmark(args)


if __name__ == '__main__':
    main()
