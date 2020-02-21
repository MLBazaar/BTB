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

* Free software: MIT license
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Documentation: https://HDI-Project.github.io/BTB
* Homepage: https://github.com/HDI-Project/BTB

# Overview

BTB ("Bayesian Tuning and Bandits") is a simple, extensible backend for developing auto-tuning
systems such as AutoML systems. It provides an easy-to-use interface for *tuning* and *selection*.

It is currently being used in several AutoML systems:
- [ATM](https://github.com/HDI-Project/ATM), distributed, multi-tenant AutoML system for
classifier tuning
- MIT TA2, MIT's system for the DARPA [Data-driven discovery of models](
https://www.darpa.mil/program/data-driven-discovery-of-models) (D3M) program
- [AutoBazaar](https://github.com/HDI-Project/AutoBazaar), a flexible, general-purpose
AutoML system

# Install

## Requirements

**BTB** has been developed and tested on [Python 3.5 and 3.6](https://www.python.org/downloads/)

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

Below there is a short example using ``BTBSession`` to perform tuning over
``ExtraTreesRegressor`` and ``RandomForestRegressor`` ensemblers from [scikit-learn](
https://scikit-learn.org/) and both of them are evaluated against the [Boston dataset](
http://lib.stat.cmu.edu/datasets/boston) regression problem.

```python3
from sklearn.datasets import load_boston as load_dataset
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from btb.session import BTBSession

models = {
    'random_forest': RandomForestRegressor,
    'extra_trees': ExtraTreesRegressor,
}

def build_model(name, hyperparameters):
    model_class = models[name]
    return model_class(random_state=0, **hyperparameters)

def score_model(name, hyperparameters):
    model = build_model(name, hyperparameters)
    r2_scorer = make_scorer(r2_score)
    scores = cross_val_score(model, X_train, y_train, scoring=r2_scorer, cv=5)
    return scores.mean()

dataset = load_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.3, random_state=0)

tunables = {
    'random_forest': {
        'n_estimators': {
            'type': 'int',
            'default': 2,
            'range': [1, 1000]
        },
        'max_features': {
            'type': 'str',
            'default': 'log2',
            'range': [None, 'auto', 'log2', 'sqrt']
        },
        'min_samples_split': {
            'type': 'int',
            'default': 2,
            'range': [2, 20]
        },
        'min_samples_leaf': {
            'type': 'int',
            'default': 2,
            'range': [1, 20]
        },
    },
    'extra_trees': {
        'n_estimators': {
            'type': 'int',
            'default': 2,
            'range': [1, 1000]
        },
        'max_features': {
            'type': 'str',
            'default': 'log2',
            'range': [None, 'auto', 'log2', 'sqrt']
        },
        'min_samples_split': {
            'type': 'int',
            'default': 2,
            'range': [2, 20]
        },
        'min_samples_leaf': {
            'type': 'int',
            'default': 2,
            'range': [1, 20]
        },
    }
}

session = BTBSession(tunables, score_model)
best_proposal = session.run(20)
```

# What's next?

For more details about **BTB** and all its possibilities and features, please check the
[project documentation site](https://HDI-Project.github.io/BTB/)!

Also do not forget to have a look at the [notebook tutorials](
https://github.com/HDI-Project/BTB/tree/master/examples/tutorials)!

# Citing BTB

If you use BTB, please consider citing our related papers.

- For the initial design and implementation of BTB (v0.1):

  Laura Gustafson. Bayesian Tuning and Bandits: An Extensible, Open Source Library for AutoML. Masters thesis, MIT EECS, June 2018. [(pdf)](https://dai.lids.mit.edu/wp-content/uploads/2018/05/Laura_MEng_Final.pdf)

  ``` bibtex
  @MastersThesis{Laura:2018,
    title = {Bayesian Tuning and Bandits: An Extensible, Open Source Library for AutoML},
    author = {Laura Gustafson},
    month = {May},
    year = {2018},
    url = {https://dai.lids.mit.edu/wp-content/uploads/2018/05/Laura_MEng_Final.pdf},
    type = {M. Eng Thesis},
    address = {Cambridge, MA},
    school = {Massachusetts Institute of Technology}",
  }
  ```

- For recent designs of BTB and its usage within the larger *ML Bazaar* project within the MIT Data to AI Lab:

  Micah J. Smith, Carles Sala, James Max Kanter, and Kalyan Veeramachaneni. ["The Machine Learning Bazaar: Harnessing the ML Ecosystem for Effective System Development."](https://arxiv.org/abs/1905.08942) arXiv Preprint 1905.08942. 2019.

  ``` bibtex
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
