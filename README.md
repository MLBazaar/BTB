<p align="left">
<img width="15%" src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="BTB" />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

![](https://raw.githubusercontent.com/HDI-Project/BTB/master/docs/images/BTB-Icon-small.png)

A simple, extensible backend for developing auto-tuning systems.

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/baytune.svg)](https://pypi.python.org/pypi/baytune)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/BTB.svg?branch=master)](https://travis-ci.org/HDI-Project/BTB)
[![Coverage Status](https://codecov.io/gh/HDI-Project/BTB/branch/master/graph/badge.svg)](https://codecov.io/gh/HDI-Project/BTB)
[![Downloads](https://pepy.tech/badge/baytune)](https://pepy.tech/project/baytune)

* License: [MIT](https://github.com/HDI-Project/BTB/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Documentation: https://HDI-Project.github.io/BTB
* Homepage: https://github.com/HDI-Project/BTB

# Overview

BTB ("Bayesian Tuning and Bandits") is a simple, extensible backend for developing auto-tuning
systems such as AutoML systems. It provides an easy-to-use interface for *tuning* and *selection*.

It is currently being used in several AutoML systems:
- [ATM](https://github.com/HDI-Project/ATM), distributed, multi-tenant AutoML system for
classifier tuning
- [mit-d3m-ta2](https://github.com/HDI-Project/mit-d3m-ta2/), MIT's system for the DARPA
[Data-driven discovery of models](https://www.darpa.mil/program/data-driven-discovery-of-models) (D3M) program
- [AutoBazaar](https://github.com/HDI-Project/AutoBazaar), a flexible, general-purpose
AutoML system

# Install

## Requirements

**BTB** has been developed and tested on [Python 3.5, 3.6 and 3.7](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **BTB** is run.

## Install with pip

The easiest and recommended way to install **BTB** is using [pip](
https://pip.pypa.io/en/stable/):

```bash
pip install baytune
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://hdi-project.github.io/BTB/contributing.html#get-started).

# Quickstart

In this short tutorial we will guide you through the necessary steps to get started using BTB
to select and tune the best model to solve a Machine Learning problem.

In particular, in this example we will be using ``BTBSession`` to perform solve the [Boston](
http://lib.stat.cmu.edu/datasets/boston) regression problem by selecting between the
`ExtraTreesRegressor` and the `RandomForestRegressor` models from [scikit-learn](
https://scikit-learn.org/) while also searching for their best Hyperparameter configuration.

## Prepare a scoring function

The first step in order to use the `BTBSession` class is to develop a scoring function.

This is a Python function that, given a model name and a hyperparameter configuration,
evaluates the performance of the model on your data and returns a score.

```python3
from sklearn.datasets import load_boston
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score

dataset = load_boston()
models = {
    'random_forest': RandomForestRegressor,
    'extra_trees': ExtraTreesRegressor,
}

def scoring_function(model_name, hyperparameter_values):
    model_class = models[model_name]
    model_instance = model_class(**hyperparameter_values)
    scores = cross_val_score(
        estimator=model_instance,
        X=dataset.data,
        y=dataset.target,
        scoring=make_scorer(r2_score)
    )
    return scores.mean()
```

## Define the tunable hyperparameters

The second step is to define the hyperparameters that we want to tune for each model as `Tunables`.

```python3
from btb.tuning import Tunable
from btb.tuning.hyperparams import CategoricalHyperParam, IntHyperParam

tunables = {
    'random_forest': Tunable({
        'max_features': CategoricalHyperParam(choices=[None, 'auto', 'log2', 'sqrt']),
        'min_samples_split': IntHyperParam(min=2, max=20, default=2),
        'min_samples_leaf': IntHyperParam(min=1, max=20, default=2)
    }),
    'extra_trees': Tunable({
        'max_features': CategoricalHyperParam(choices=[None, 'auto', 'log2', 'sqrt']),
        'min_samples_split': IntHyperParam(min=2, max=20, default=2),
        'min_samples_leaf': IntHyperParam(min=1, max=20, default=2)
    })
}
```

## Start the searching process

Once you have defined a scoring function and the tunable hyperparameters specification of your
models, you can start the searching for the best model and hyperparameter configuration by using
the `btb.BTBSession`.

All you need to do is create an instance passing the tunable hyperparameters scpecification
and the scoring function.

```python3
from btb import BTBSession

session = BTBSession(
    tunables=tunables,
    scorer=scoring_function
)
```

And then call the `run` method indicating how many tunable iterations you want the Session to
perform:


```python3
best_proposal = session.run(20)
```

The result will be a dictionary indicating the name of the best model that could be found
and the hyperparameter configuration that was used:

```
{
    'id': 'd85262197592bd00c8cd9e87164e18c8',
    'name': 'extra_trees',
    'config': {
        'max_features': None,
        'min_samples_split': 17,
        'min_samples_leaf': 1
    },
    'score': 0.6056926625119803
}
 ```

# How does BTB perform?

We have a comprehensive [benchmarking framework](https://github.com/HDI-Project/BTB/tree/master/benchmark)
that we use to evaluate the performance of our `Tuners`. For every release, we perform benchmarking
against 100's of challenges, comparing tuners against each other in terms of number of wins.
We present the latest leaderboard from latest release below:

## Number of Wins per Version

| tuner                   | with ties | without ties |
|-------------------------|-----------|--------------|
| `BTB.GPEiTuner`         |    **35** |            7 |
| `BTB.GPTuner`           |    33     |        **8** |
| `BTB.UniformTuner`      |    29     |            2 |
| `HyperOpt.rand.suggest` |    28     |            0 |
| `HyperOpt.tpe.suggest`  |    32     |            5 |

- Detailed results from which this summary emerged are available [here](https://docs.google.com/spreadsheets/d/1E0fSSfqOuDhazccdsx7eG1aLCJagdpj1OKYhdOohZOg/).
- If you want to compare your own tuner, follow the steps in our benchmarking framework [here](https://github.com/HDI-Project/BTB/tree/master/benchmark).
- If you have a proposal for tuner that we should include in our benchmarking get in touch
with us at [dailabmit@gmail.com](mailto:dailabmit@gmail.com).

> :warning: **Note**: In release v0.3.7, we are currently only doing 50 ML challenges. Our next release for
benchmarking will have results from 422 datasets and 3 optimization challenges. To check out what
will be included in our benchmarking efforts - you can check [here](https://github.com/HDI-Project/BTB/projects).

# What's next?

For more details about **BTB** and all its possibilities and features, please check the
[project documentation site](https://HDI-Project.github.io/BTB/)!

Also do not forget to have a look at the [notebook tutorials](notebooks).

# Citing BTB

If you use BTB, please consider citing our related paper:

  Micah J. Smith, Carles Sala, James Max Kanter, and Kalyan Veeramachaneni. ["The Machine Learning Bazaar: Harnessing the ML Ecosystem for Effective System Development."](https://arxiv.org/abs/1905.08942) arXiv Preprint 1905.08942. 2019.

  ```bibtex
  @article{smith2019mlbazaar,
    author = {Smith, Micah J. and Sala, Carles and Kanter, James Max and Veeramachaneni, Kalyan},
    title = {The Machine Learning Bazaar: Harnessing the ML Ecosystem for Effective System Development},
    journal = {arXiv e-prints},
    year = {2019},
    eid = {arXiv:1905.08942},
    pages = {arXiv:1905.08942},
    archivePrefix = {arXiv},
    eprint = {1905.08942},
  }
  ```
