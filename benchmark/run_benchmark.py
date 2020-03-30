import argparse
import logging
import random
import warnings
from datetime import datetime

import tabulate

from btb.benchmark import benchmark
from btb.benchmark.challenges import MATH_CHALLENGES, RandomForestChallenge
from btb.benchmark.tuners import get_all_tuning_functions

LOGGER = logging.getLogger(__name__)
ALL_TYPES = ['math', 'random_forest']

warnings.filterwarnings("ignore")


def _get_math_challenge(challenge):
    if isinstance(challenge, str):
        for math_challenge in MATH_CHALLENGES:
            if str(challenge).lower() == str(math_challenge.__name__).lower():
                return math_challenge

    else:
        return challenge()


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


CHALLENGES = {
    'math': _get_math_challenge,
    'random_forest': RandomForestChallenge,
}


def _get_all_challenges(args):
    all_challenges = []
    if args.type is None:
        return RandomForestChallenge.get_available_datasets() + MATH_CHALLENGES

    if 'math' in args.type:
        all_challenges = all_challenges + MATH_CHALLENGES
    if 'random_forest' in args.type:
        all_challenges = all_challenges + RandomForestChallenge.get_available_datasets()

    return all_challenges


def _get_challenges(args):
    challenges = args.challenges or _get_all_challenges(args)
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
                challenge = CHALLENGES[challenge_type](challenge_name)
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
    results = benchmark(candidates, challenges, args.iterations, args.complete_dataframe)

    if args.report is None:
        args.report = datetime.now().strftime('benchmark_%Y%m%d%H%M') + '.csv'

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
                        choices=['math', 'random_forest'])
    parser.add_argument('-c', '--complete-dataframe', action='store_true',
                        help='Return the complete dataframe with additional information.')

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
