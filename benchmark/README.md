# Overview

This folder contains python scripts that exceute the `benchmark` module from `BTB` in order to
generate a leaderboard. This benchmarking process consists on scoring the `tuner`'s proposed
hyperparameters for a given `challenge` in a certain amount of iterations.

## The Benchmarking process

**BTB** comes with a `benchmark` package that allows users to run a performance test on
their tuners. This benchmark process iterates over a collection of `challenges` and executes a
`tuner_function` for each one of the `challenges` for a given amount of iterations.

Once this process is done, a `dataframe` is being returned with the following information:
- Challenge name.
- Tuner name.
- Score.

# Tuners

The goal of this benchmarking process is to score any tuner or library that shares the same focus
as `BTB`: Bayesian tuning or hyperparameter optimization. At the moment our benchmarking process
is scoring the following libraries:

- [BTB](https://github.com/HDI-Project/BTB) Bayesian Tuning and Bandits.
- [Hyperopt](https://github.com/hyperopt/hyperopt) Hyperopt: Distributed Hyperparameter Optimization.

# Challenges

The benchmarking process computes scores over two types of challenges:

- Math Challenges: Mathematical functions that compute a score regarding some input parameters.
- Machine Learning Challenge: An estimator that is being tuned in order to improve the obtained
score for a given dataset.


# Exectution

The benchmarking process is being launched for every release of
[BTB](https://github.com/HDI-Project/BTB) against more than 400 challenges. All previous executions
can be found inside the `results` folder located
[here](https://github.com/HDI-Project/BTB/tree/master/benchmark/results). After every execution our
`leaderboard` is being updated with a short summary where we compute on how many challenges a
`tuner` has obtained the best score.
