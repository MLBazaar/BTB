import argparse
import logging
import random
from datetime import datetime

import tabulate

from btb.benchmark import DEFAULT_CHALLENGES, ATMChallenge, benchmark, get_all_tuning_functions

LOGGER = logging.getLogger(__name__)


def perform_benchmark(args):
    candidates = get_all_tuning_functions()
    challenges = []

    if args.more_challenges:
        challenges = [challenge_class() for challenge_class in DEFAULT_CHALLENGES]

    if args.sample:
        LOGGER.info('Randomly selecting %s datasets', args.sample)
        available_datasets = ATMChallenge.get_available_datasets()
        selected_datasets = random.choices(available_datasets, k=args.sample)
        challenges.extend(ATMChallenge.get_all_challenges(challenges=selected_datasets))

        challenges = random.choices(challenges, k=args.sample)

    elif args.challenges:
        challenges = ATMChallenge.get_all_challenges(challenges=args.challenges)

    else:
        LOGGER.info('Loading all the datasets available.')
        challenges = ATMChallenge.get_all_challenges()

    results = benchmark(candidates, challenges, args.iterations)

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
    # Parser
    parser = argparse.ArgumentParser(description='BTB Benchmark Command Line Interface')

    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')
    parser.add_argument('-m', '--more-challenges', action='store_true',
                        help='Include all the challenges that are found in BTB.')
    parser.add_argument('-r', '--report', type=str, required=False,
                        help='Path to the CSV file where the report will be dumped')
    parser.add_argument('-s', '--sample', type=int,
                        help='Limit the test to a sample of datasets for the given size.')
    parser.add_argument('-i', '--iterations', type=int, default=100,
                        help='Number of iterations to perform per challenge with each candidate.')
    parser.add_argument('challenges', nargs='*', help='Name of the challenge/s to be processed.')

    return parser


def logging_setup(verbosity=1, logfile=None, logger_name=None, stdout=True):
    logger = logging.getLogger(logger_name)
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout or not logfile:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def main():
    parser = _get_parser()
    args = parser.parse_args()

    logging_setup(args.verbose)

    perform_benchmark(args)


if __name__ == '__main__':
    main()
