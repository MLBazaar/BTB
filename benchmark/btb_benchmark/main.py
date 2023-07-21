# -*- coding: utf-8 -*-
import logging
import random
import socket
from datetime import datetime, timedelta

import dask
import pandas as pd
from distributed.client import futures_of
from distributed.diagnostics.progressbar import TextProgressBar

from baytune.tuning.tuners.base import BaseTuner
from btb_benchmark.challenges import (
    MATH_CHALLENGES,
    RandomForestChallenge,
    SGDChallenge,
    XGBoostChallenge,
)
from btb_benchmark.challenges.challenge import Challenge
from btb_benchmark.challenges.datasets import get_dataset_names
from btb_benchmark.results import load_results, write_results
from btb_benchmark.tuning_functions import get_all_tuning_functions
from btb_benchmark.tuning_functions.btb import make_btb_tuning_function

LOGGER = logging.getLogger(__name__)
ALL_TYPES = ["math", "xgboost"]


def get_math_challenge_instance(name, *args, **kwargs):
    return MATH_CHALLENGES.get(name)()


CHALLENGE_GETTER = {
    "math": get_math_challenge_instance,
    "random_forest": RandomForestChallenge,
    "sgd": SGDChallenge,
    "xgboost": XGBoostChallenge,
}


@dask.delayed
def _evaluate_tuner_on_challenge(name, tuner, challenge, iterations):
    tunable_hyperparameters = challenge.get_tunable_hyperparameters()
    LOGGER.info("Evaluating tuner %s on challenge %s for %s iterations",
                name, challenge, iterations)
    try:
        start = datetime.utcnow()
        score = tuner(challenge.evaluate, tunable_hyperparameters, iterations)
        result = {
            "challenge": str(challenge),
            "tuner": name,
            "score": score,
            "iterations": iterations,
            "elapsed": datetime.utcnow() - start,
            "hostname": socket.gethostname()
        }
        if hasattr(challenge, "data"):
            result["rows"] = challenge.data[0].shape[0]

    except Exception as ex:
        LOGGER.warn(
            "Could not score tuner %s with challenge %s, error: %s", name, challenge, ex)
        result = {
            "challenge": str(challenge),
            "tuner": name,
            "score": None,
            "elapsed": datetime.utcnow() - start,
            "hostname": socket.gethostname()
        }

    return result


def _evaluate_tuners_on_challenge(tuners, challenge, iterations):
    LOGGER.info("Evaluating challenge %s", challenge)
    results = []
    for name, tuner in tuners.items():
        try:
            result = _evaluate_tuner_on_challenge(name, tuner, challenge, iterations)
            results.append(result)
        except Exception as ex:
            LOGGER.warn(
                "Could not score tuner %s with challenge %s, error: %s", name, challenge, ex)

    return results


class LogProgressBar(TextProgressBar):
    last = 0
    logger = logging.getLogger("distributed")

    def _draw_bar(self, remaining, all, **kwargs):
        frac = (1 - remaining / all) if all else 0

        if frac > self.last + 0.01:
            self.last = int(frac * 100) / 100
            bar = "#" * int(self.width * frac)
            percent = int(100 * frac)

            time_per_task = self.elapsed / (all - remaining)
            remaining_time = timedelta(seconds=time_per_task * remaining)
            eta = datetime.utcnow() + remaining_time

            elapsed = timedelta(seconds=self.elapsed)
            msg = "[{0:<{1}}] | {2}% Completed | {3} | {4} | {5}".format(
                bar, self.width, percent, elapsed, remaining_time, eta
            )
            self.logger.info(msg)

    def _draw_stop(self, **kwargs):
        pass


def progress(*futures):
    futures = futures_of(futures)
    if not isinstance(futures, (set, list)):
        futures = [futures]

    LogProgressBar(futures)


def benchmark(tuners, challenges, iterations, detailed_output=False):
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
                * tunable (baytune.tuning.Tunable):
                    A ``Tunable`` instance used to instantiate a tuner.
                * iterations (int):
                    Number of tuning iterations to perform.

        challenges (list):
            A list of ``chalenges``. This challenges must inherit from
            ``baytune.challenges.challenge.Challenge``.
        iterations (int):
            Amount of tuning iterations to perform for each tuner and each challenge.
        detailed_output (bool):
            If ``True`` a dataframe with the elapsed time, score and iterations will be returned.

    Returns:
        pandas.DataFrame:
            A ``pandas.DataFrame`` with the obtained scores for the given challenges is being
            returned.
    """
    delayed = []

    for challenge in challenges:
        result = _evaluate_tuners_on_challenge(tuners, challenge, iterations)
        delayed.extend(result)

    persisted = dask.persist(*delayed)

    try:
        progress(persisted)
    except ValueError:
        # Using local client. No progress bar needed.
        pass

    results = dask.compute(*persisted)

    df = pd.DataFrame.from_records(results)
    if detailed_output:
        return df

    df = df.pivot(index="challenge", columns="tuner", values="score")
    df.index.rename(None, inplace=True)
    df.columns.rename(None, inplace=True)

    return df


def _as_list(param):
    """Make sure that param is either ``None`` or a ``list``."""
    if param is None or isinstance(param, (list, tuple)):
        return param

    return [param]


def _challenges_as_list(param):
    """Make sure that param is either ``None`` or a ``list``."""
    if param is None or isinstance(param, (list, tuple)):
        return param

    return get_dataset_names(param)


def _get_tuners_dict(tuners=None):
    all_tuners = get_all_tuning_functions()
    if tuners is None:
        LOGGER.info("Using all tuning functions.")
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
                    LOGGER.info("Loading tuning function: %s", tuner)
                    selected_tuning_functions[tuner] = tuning_function
                else:
                    LOGGER.info("Could not load tuning function: %s", tuner)

        if not selected_tuning_functions:
            raise ValueError("No tunable function was loaded.")

        return selected_tuning_functions


def _get_all_challenge_names(challenge_types=None):
    all_challenge_names = []
    if "math" in challenge_types:
        all_challenge_names += list(MATH_CHALLENGES.keys())
    if any(name in challenge_types for name in ("sdg", "xgboost", "random_forest")):
        all_challenge_names += get_dataset_names("all")

    return all_challenge_names


def _get_challenges_list(challenges=None, challenge_types=None, sample=None, max_rows=None):
    challenge_types = _as_list(challenge_types) or ALL_TYPES
    challenges = _challenges_as_list(challenges) or _get_all_challenge_names(challenge_types)

    selected = []
    unknown = []

    if sample:
        if sample > len(challenges):
            raise ValueError("Sample can not be greater than {}".format(len(challenges)))

        challenges = random.sample(challenges, sample)

    for challenge in challenges:
        known = False
        if isinstance(challenge, Challenge):
            selected.append(challenge)
        else:
            for challenge_type in challenge_types:
                try:
                    challenge_class = CHALLENGE_GETTER[challenge_type]
                    challenge_instance = challenge_class(challenge, max_rows=max_rows)

                    if challenge_instance:
                        known = True
                        selected.append(challenge_instance)
                except Exception:
                    pass

            if not known:
                unknown.append(challenge)

    if unknown:
        raise ValueError("Challenges {} not of type {}".format(unknown, challenge_types))

    if not selected:
        raise ValueError("No challenges selected!")

    return selected


def run_benchmark(tuners=None, challenge_types=None, challenges=None, sample=None,
                  iterations=100, max_rows=None, output_path=None, detailed_output=False):
    """Execute the benchmark function and optionally store the result as a ``CSV``.

    This function provides a user-friendly interface to interact with the ``benchmark``
    function. It allows the user to specify an ``output_path`` where the results can be
    stored. If this path is not provided, a ``pandas.DataFrame`` will be returned.

    Args:
        tuners (str, baytune.tuning.tuners.base.BaseTuner or list):
            Tuner name, ``baytune.tuning.tuners.base.BaseTuner`` subclass or a list with the previously
            described objects. If ``None`` all available ``tuners`` implemented in
            ``btb_benchmark`` will be used.
        challenge_types (str or list):
            Type or list of types for challenges to be benchmarked, if ``None`` all available
            types will be used.
        challenges (str or list):
            If ``str`` it will be interpreted as ``collection`` of datasets (currently: all or
            openml100). A list containing: challenge name, ``btb_benchmark.challenge.Challenge``
            instance or a list with the previously described objects. If ``None`` will use
            ``challenge_types`` to determine which challenges to use.
        sample (int):
            Run only on a subset of the available datasets of the given size.
        iterations (int):
            Number of tuning iterations to perform per challenge and tuner.
        max_rows (int):
            Maximum number of rows to use from each dataset. If ``None``, or if the
            given number is higher than the number of rows in the dataset, the entire
            dataset is used. Defaults to ``None``.
        output_path (str):
            If an ``output_path`` is given, the final results will be saved in that location.
        detailed_output (bool):
            If ``True`` a dataframe with the elapsed time, score and iterations will be returned.

    Returns:
        pandas.DataFrame or None:
            If ``output_path`` is ``None`` it will return a ``pandas.DataFrame`` object,
            else it will dump the results in the specified ``output_path``.
    """
    tuners = _get_tuners_dict(tuners)
    challenges = _get_challenges_list(
        challenges=challenges,
        challenge_types=challenge_types,
        sample=sample,
        max_rows=max_rows
    )

    results = benchmark(tuners, challenges, iterations, detailed_output)

    if output_path:
        LOGGER.info("Saving benchmark report to %s", output_path)
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
