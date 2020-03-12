# -*- coding: utf-8 -*-
import argparse
import logging

import pandas as pd
import tabulate

from btb.benchmark.challenges import MATH_CHALLENGES, ML_CHALLENGES
from btb.benchmark.challenges.challenge import ATMChallenge
from btb.benchmark.tuners import get_all_tuning_functions

DEFAULT_CHALLENGES = MATH_CHALLENGES + ML_CHALLENGES
LOGGER = logging.getLogger(__name__)


def evaluate_candidate(name, candidate, challenges, iterations):
    candidate_result = []

    if not isinstance(challenges, list):
        challenges = [challenges]

    for challenge in challenges:
        tunable_hyperparameters = challenge.get_tunable_hyperparameters()

        try:
            score = candidate(challenge.evaluate, tunable_hyperparameters, iterations)
            result = {
                'challenge': str(challenge),
                'candidate': name,
                'score': score,
            }

        except Exception as ex:
            LOGGER.warn(
                'Could not score candidate %s with challenge %s, error: %s', name, challenge, ex)

        if result:
            candidate_result.append(result)

    return candidate_result


def benchmark(candidates, challenges=None, iterations=1000):
    """Benchmark function.

    This benchmark function iterates over a collection of ``challenges`` and executes a
    ``tuner_function`` for each one of the ``challenges`` for a given amount of iterations.

    Args:
        candidates (callable, list, tuple or dict):
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
        challenges = [challenge_class() for challenge_class in DEFAULT_CHALLENGES]

    if callable(candidates):
        candidates = {candidates.__name__: candidates}

    elif isinstance(candidates, (list, tuple)):
        candidates = {candidate.__name__: candidate for candidate in candidates}

    elif not isinstance(candidates, dict):
        raise TypeError(
            'Candidates can only be a callable, list of callables, tuple of callables or dict.')

    if not isinstance(challenges, list):
        challenges = [challenges]

    results = []

    for name, candidate in candidates.items():

        result = evaluate_candidate(name, candidate, challenges, iterations)

        if result:
            results.extend(result)

    df = pd.DataFrame.from_records(results)
    df = df.pivot(index='candidate', columns='challenge', values='score')

    del df.columns.name
    del df.index.name

    return df


def _load_candidates(args):
    all_candidates = get_all_tuning_functions()

    if args.all:
        return all_candidates

    if args.candidate:
        candidates = {}
        for candidate in args.candidate:
            try:
                candidates[candidate] = all_candidates[candidate]
            except Exception:
                LOGGER.warn('Could not load candidate %s', candidate)

        return candidates


def evaluate(challenges, args):
    """
    Evaluate from Command Line Inputs.
    """
    candidates = _load_candidates(args)
    results = benchmark(candidates, challenges=challenges, iterations=args.iterations)

    print(tabulate.tabulate(
        results,
        showindex=False,
        tablefmt='github',
        headers=results.columns
    ))

    if args.report:
        results.to_csv(args.report, index=False)


def _atmchallenge_benchmark(args):

    if args.all:
        challenges = ATMChallenge.get_all_challenges()

    elif args.challenges:
        challenges = ATMChallenge.get_all_challenges(challenges=args.challenges)

    else:
        raise ValueError('No challenge dataset provided.')

    evaluate(challenges, args)


def _standard_benchmark(args):
    if args.all:
        challenges = DEFAULT_CHALLENGES

    elif args.challenges:
        challenges = {}
        available_challenges = {challenge.__name__: challenge for challenge in DEFAULT_CHALLENGES}

        for challenge in args.challenges:
            try:
                challenges[challenge] = available_challenges[challenge]
            except Exception:
                LOGGER.warn('Could not load challenge %s', challenge)

    evaluate(challenges, args)


def _get_parser():

    # Common parsers
    report = argparse.ArgumentParser(add_help=False)
    report.add_argument('-r', '--report', type=str, required=False,
                        help='Path to the CSV file where the report will be dumped')

    challenges_args = argparse.ArgumentParser(add_help=False)
    challenges_args.add_argument('-a', '--all', action='store_true',
                                 help='Process all the challenges available for the given mode.')
    challenges_args.add_argument('challenge', nargs='*',
                                 help='Name of the challenge/s to be processed.')
    challenges_args.add_argument(
        '-i',
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations to perform for each challenge with each candidate.'
    )

    candidates_args = argparse.ArgumentParser(add_help=False)
    candidates_args.add_argument('-c', '--candidates', action='store_true',
                                 help='Use all the available tuners as candidates.')
    candidates_args.add_argument('candidate', nargs='*',
                                 help='Name of the candidate / candidates to use')

    # Parser
    parser = argparse.ArgumentParser(
        description='BTB Benchmark Command Line Interface',
        parents=[report, challenges_args, candidates_args]
    )
    subparsers = parser.add_subparsers(title='mode', dest='mode', help='Mode of operation.')
    subparsers.requiered = True

    parser.set_defaults(mode=None)

    # ATMChallenge Mode
    atmchallenge_args = subparsers.add_parser(
        'atm',
        parents=[report, challenges_args, candidates_args],
        help='Perform benchmark with ATMChallenges.'
    )
    atmchallenge_args.set_defaults(mode=_atmchallenge_benchmark)

    # Standard Mode
    standard_mode = subparsers.add_parser(
        'standard',
        parents=[report, challenges_args, candidates_args],
        help='Perform benchmark with Challenges or MLChallenges'
    )

    standard_mode.set_defaults(mode=_standard_benchmark)

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    if not args.mode:
        parser.print_help()
        parser.exit()

    args.mode(args)


if __name__ == '__main__':
    main()
