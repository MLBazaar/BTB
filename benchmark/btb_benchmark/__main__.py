# -*- coding: utf-8 -*-
import argparse
import logging
import sys
import warnings

import tabulate

from btb_benchmark.main import run_benchmark, summarize_results


def _run(args):
    # Logger setup
    log_level = (3 - args.verbose) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(level=log_level, format=fmt)
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("hyperopt").setLevel(logging.ERROR)
    logging.getLogger("ax").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # run
    results = run_benchmark(
        args.tuners,
        args.challenge_types,
        args.challenges,
        args.sample,
        args.iterations,
        args.max_rows,
        args.output_path,
        args.detailed_output,
    )

    if not args.output_path:
        print(tabulate.tabulate(
            results,
            tablefmt='github',
            headers=results.columns
        ))


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
    run.add_argument('-m', '--max-rows', type=int,
                     help='Max amount of rows to use for each dataset.')
    run.add_argument('-d', '--detailed-output', action='store_true',
                     help='Output a detailed dataset with elapsed times.')

    # Summarize action
    summary = action.add_parser('summary', help='Summarize the BTB Benchmark results')
    summary.set_defaults(action=_summary)
    summary.add_argument('input', nargs='+', help='Input path with results.')
    summary.add_argument('output', help='Output file.')

    return parser


def main():
    warnings.filterwarnings("ignore")

    # Parse args
    parser = _get_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    args.action(args)


if __name__ == '__main__':
    main()
