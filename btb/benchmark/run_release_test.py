import argparse
import logging
from datetime import datetime

import tabulate

from btb.benchmark import ATMChallenge, benchmark, get_all_tuning_functions

LOGGER = logging.getLogger(__name__)


def perform_benchmark(args):
    candidates = get_all_tuning_functions()

    if args.challenges:
        challenges = ATMChallenge.get_all_challenges(challenges=args.challenges)
    else:
        challenges = ATMChallenge.get_all_challenges()

    results = benchmark(candidates, challenges, args.iterations)

    if args.report is None:
        args.report = str(datetime.timestamp(datetime.now())) + '.csv'

    LOGGER.info('Saving benchmark report to %s', args.report)

    print('\n')
    print(tabulate.tabulate(
        results,
        tablefmt='github',
        headers=results.columns
    ))

    results.to_csv(args.report)


def _get_parser():
    # Common parsers
    report = argparse.ArgumentParser(add_help=False)
    report.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')
    report.add_argument('-r', '--report', type=str, required=False,
                        help='Path to the CSV file where the report will be dumped')

    challenges_args = argparse.ArgumentParser(add_help=False)
    challenges_args.add_argument('challenges', nargs='*',
                                 help='Name of the challenge/s to be processed.')
    challenges_args.add_argument(
        '-i',
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations to perform for each challenge with each candidate.'
    )

    # Parser
    parser = argparse.ArgumentParser(
        description='BTB Benchmark Command Line Interface',
        parents=[report, challenges_args]
    )

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
