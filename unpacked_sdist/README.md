<p align="left">
<img width="15%" src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="BTB" />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

![](https://raw.githubusercontent.com/MLBazaar/BTB/master/docs/images/BTB-Icon-small.png)

A simple, extensible backend for developing auto-tuning systems.

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/baytune.svg)](https://pypi.python.org/pypi/baytune)
[![Travis CI Shield](https://travis-ci.com/MLBazaar/BTB.svg?branch=master)](https://travis-ci.com/MLBazaar/BTB)
[![Coverage Status](https://codecov.io/gh/MLBazaar/BTB/branch/master/graph/badge.svg)](https://codecov.io/gh/MLBazaar/BTB)
[![Downloads](https://pepy.tech/badge/baytune)](https://pepy.tech/project/baytune)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MLBazaar/BTB/master?filepath=tutorials)

* License: [MIT](https://github.com/MLBazaar/BTB/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Documentation: https://mlbazaar.github.io/BTB
* Homepage: https://github.com/MLBazaar/BTB

# Overview

BTB ("Bayesian Tuning and Bandits") is a simple, extensible backend for developing auto-tuning
systems such as AutoML systems. It provides an easy-to-use interface for *tuning* models and
*selecting* between models.

It is currently being used in several AutoML systems:

- [ATM](https://github.com/HDI-Project/ATM), a distributed, multi-tenant AutoML system for
classifier tuning
- [MIT's system](https://github.com/HDI-Project/mit-d3m-ta2/) for the DARPA
[Data-driven discovery of models](https://www.darpa.mil/program/data-driven-discovery-of-models) (D3M) program
- [AutoBazaar](https://github.com/MLBazaar/AutoBazaar), a flexible, general-purpose
AutoML system

## Try it out now!

If you want to quickly discover **BTB**, simply click the button below and follow the tutorials!

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MLBazaar/BTB/master?filepath=tutorials)

# Install

## Requirements

**BTB** has been developed and tested on [Python 3.6, 3.7 and 3.8](https://www.python.org/downloads/)

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
[Contributing Guide](https://mlbazaar.github.io/BTB/contributing.html#get-started).

# Quickstart

In this short tutorial we will guide you through the necessary steps to get started using BTB
to `select` between models and `tune` a model to solve a Machine Learning problem.

In particular, in this example we will be using ``BTBSession`` to perform solve the [Wine](
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data) classification problem
by selecting between the `DecisionTreeClassifier` and the `SGDClassifier` models from
[scikit-learn](https://scikit-learn.org/) while also searching for their best `hyperparameter`
configuration.

## Prepare a scoring function

The first step in order to use the `BTBSession` class is to develop a `scoring` function.

This is a Python function that, given a model name and a `hyperparameter` configuration,
evaluates the performance of the model on your data and returns a score.

```python3
from sklearn.datasets import load_wine
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


dataset = load_wine()
models = {
    'DTC': DecisionTreeClassifier,
    'SGDC': SGDClassifier,
}

def scoring_function(model_name, hyperparameter_values):
    model_class = models[model_name]
    model_instance = model_class(**hyperparameter_values)
    scores = cross_val_score(
        estimator=model_instance,
        X=dataset.data,
        y=dataset.target,
        scoring=make_scorer(f1_score, average='macro')
    )
    return scores.mean()
```

## Define the tunable hyperparameters

The second step is to define the `hyperparameters` that we want to `tune` for each model as
`Tunables`.

```python3
from btb.tuning import Tunable
from btb.tuning import hyperparams as hp

tunables = {
    'DTC': Tunable({
        'max_depth': hp.IntHyperParam(min=3, max=200),
        'min_samples_split': hp.FloatHyperParam(min=0.01, max=1)
    }),
    'SGDC': Tunable({
        'max_iter': hp.IntHyperParam(min=1, max=5000, default=1000),
        'tol': hp.FloatHyperParam(min=1e-3, max=1, default=1e-3),
    })
}
```

## Start the searching process

Once you have defined a `scoring` function and the tunable `hyperparameters` specification of your
models, you can start the searching for the best model and `hyperparameter` configuration by using
the `btb.BTBSession`.

All you need to do is create an instance passing the tunable `hyperparameters` scpecification
and the scoring function.

```python3
from btb import BTBSession

session = BTBSession(
    tunables=tunables,
    scorer=scoring_function
)
```

And then call the `run` method indicating how many tunable iterations you want the `BTBSession` to
perform:


```python3
best_proposal = session.run(20)
```

The result will be a dictionary indicating the name of the best model that could be found
and the `hyperparameter` configuration that was used:

```
{
    'id': '826aedc2eff31635444e8104f0f3da43',
    'name': 'DTC',
    'config': {
        'max_depth': 21,
        'min_samples_split': 0.044010284821858835
    },
    'score': 0.907229308339589
}
 ```

# How does BTB perform?

We have a comprehensive [benchmarking framework](https://github.com/MLBazaar/BTB/tree/master/benchmark)
that we use to evaluate the performance of our `Tuners`. For every release, we perform benchmarking
against 100's of challenges, comparing tuners against each other in terms of number of wins.
We present the latest leaderboard from latest release below:

## Number of Wins on latest Version

| tuner                   | with ties | without ties |
|-------------------------|-----------|--------------|
| `Ax.optimize`           |    220    |           32 |
| `BTB.GCPEiTuner`        |    139    |            2 |
| `BTB.GCPTuner`          |  **252**  |       **90** |
| `BTB.GPEiTuner`         |    208    |           16 |
| `BTB.GPTuner`           |    213    |           24 |
| `BTB.UniformTuner`      |    177    |            1 |
| `HyperOpt.tpe`          |    186    |            6 |
| `SMAC.HB4AC`            |    180    |            4 |
| `SMAC.SMAC4HPO_EI`      |    220    |           31 |
| `SMAC.SMAC4HPO_LCB`     |    205    |           16 |
| `SMAC.SMAC4HPO_PI`      |    221    |           35 |

- Detailed results from which this summary emerged are available [here](https://docs.google.com/spreadsheets/d/15a-pAV_t7CCDvqDyloYmdVNFhiKJFOJ7bbgpmYIpyTs/edit?usp=sharing).
- If you want to compare your own tuner, follow the steps in our benchmarking framework [here](https://github.com/MLBazaar/BTB/tree/master/benchmark).
- If you have a proposal for tuner that we should include in our benchmarking get in touch
with us at [dailabmit@gmail.com](mailto:dailabmit@gmail.com).

# More tutorials

1. To just `tune` `hyperparameters` - see our `tuning` tutorial [here](
https://github.com/MLBazaar/BTB/blob/master/tutorials/01_Tuning.ipynb) and
[documentation here](https://mlbazaar.github.io/BTB/tutorials/01_Tuning.html).
2. To see the [types of `hyperparameters`](
https://mlbazaar.github.io/BTB/tutorials/01_Tuning.html#What-is-a-Hyperparameter?) we support
see our [documentation here](https://mlbazaar.github.io/BTB/tutorials/01_Tuning.html#What-is-a-Hyperparameter?).
3. You can read about [our benchmarking framework here](https://mlbazaar.github.io/BTB/benchmark.html#).
4. See our [tutorial on `selection` here](https://github.com/MLBazaar/BTB/blob/master/tutorials/02_Selection.ipynb)
and [documentation here](https://mlbazaar.github.io/BTB/tutorials/02_Selection.html).

For more details about **BTB** and all its possibilities and features, please check the
[project documentation site](https://mlbazaar.github.io/BTB/)!

Also do not forget to have a look at the [notebook tutorials](tutorials).

# Citing BTB

If you use **BTB**, please consider citing the following [paper](
https://arxiv.org/pdf/1905.08942.pdf):

```
@article{smith2019mlbazaar,
  author = {Smith, Micah J. and Sala, Carles and Kanter, James Max and Veeramachaneni, Kalyan},
  title = {The Machine Learning Bazaar: Harnessing the ML Ecosystem for Effective System Development},
  journal = {arXiv e-prints},
  year = {2019},
  eid = {arXiv:1905.08942},
  pages = {arxiv:1904.09535},
  archivePrefix = {arXiv},
  eprint = {1905.08942},
}
`````
