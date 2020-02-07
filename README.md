<p align="left">
<img width="15%" src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="BTB" />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

![](https://raw.githubusercontent.com/HDI-Project/BTB/master/docs/images/BTB-Icon-small.png)

A simple, extensible backend for developing auto-tuning systems.

[![PyPi Shield](https://img.shields.io/pypi/v/baytune.svg)](https://pypi.python.org/pypi/baytune)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/BTB.svg?branch=master)](https://travis-ci.org/HDI-Project/BTB)
[![Coverage Status](https://codecov.io/gh/HDI-Project/BTB/branch/master/graph/badge.svg)](https://codecov.io/gh/HDI-Project/BTB)
[![Downloads](https://pepy.tech/badge/baytune)](https://pepy.tech/project/baytune)

* Free software: MIT license
* Documentation: https://HDI-Project.github.io/BTB
* Homepage: https://github.com/HDI-Project/BTB

# Overview

BTB ("Bayesian Tuning and Bandits") is a simple, extensible backend for developing auto-tuning systems such as AutoML systems. It provides an easy-to-use interface for *tuning* and *selection*.

It is currently being used in several AutoML systems:
- [ATM](https://github.com/HDI-Project/ATM), distributed, multi-tenant AutoML system for classifier tuning
- MIT TA2, MIT's system for the DARPA [Data-driven discovery of models](https://www.darpa.mil/program/data-driven-discovery-of-models) (D3M) program
- [AutoBazaar](https://github.com/HDI-Project/AutoBazaar), a flexible, general-purpose AutoML system

# Quickstart

## Install

BTB has been developed and tested on [Python 3.5, 3.6, and 3.7](https://www.python.org/downloads).

```bash
pip install baytune
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).

## Tuners

Tuners are specifically designed to speed up the process of selecting the
optimal hyper parameter values for a specific machine learning algorithm.

`btb.tuning.tuners` defines Tuners: classes with a fit/predict/propose interface for
suggesting sets of hyperparameters.

This is done by following a Bayesian Optimization approach and iteratively:

* letting the tuner propose new sets of hyper parameter
* fitting and scoring the model with the proposed hyper parameters
* passing the score obtained back to the tuner

At each iteration the tuner will use the information already obtained to propose
the set of hyper parameters that it considers that have the highest probability
to obtain the best results.

To instantiate a ``Tuner`` all we need is a ``Tunable`` class with a collection of
``hyperparameters``.

``` python
>>> from btb.tuning import Tunable
>>> from btb.tuning.tuners import GPTuner
>>> from btb.tuning.hyperparams import IntHyperParam
>>> hyperparams = {
...     'n_estimators': IntHyperParam(min=10, max=500),
...     'max_depth': IntHyperParam(min=10, max=500),
... }
>>> tunable = Tunable(hyperparams)
>>> tuner = GPTuner(tunable)
```

Then we perform the following three steps in a loop.

1. Let the Tuner propose a new set of parameters:

    ``` python
    >>> parameters = tuner.propose()
    >>> parameters
    {'n_estimators': 297, 'max_depth': 3}
    ```

2. Fit and score a new model using these parameters:

    ``` python
    >>> model = RandomForestClassifier(**parameters)
    >>> model.fit(X_train, y_train)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=3, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=297, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    >>> score = model.score(X_test, y_test)
    >>> score
    0.77
    ```

3. Pass the used parameters and the score obtained back to the tuner:

    ``` python
    tuner.record(parameters, score)
    ```

At each iteration, the ``Tuner`` will use the information about the previous tests
to evaluate and propose the set of parameter values that have the highest probability
of obtaining the highest score.

### Selectors

The selectors are intended to be used in combination with tuners in order to find
out and decide which model seems to get the best results once it is properly fine tuned.

In order to use the selector we will create a ``Tuner`` instance for each model that
we want to try out, as well as the ``Selector`` instance.

```python
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.svm import SVC
>>> from btb.selection import UCB1
>>> from btb.tuning.hyperparams import FloatHyperParam
>>> models = {
...     'RF': RandomForestClassifier,
...     'SVC': SVC
... }
>>> selector = UCB1(['RF', 'SVC'])
>>> rf_hyperparams = {
...     'n_estimators': IntHyperParam(min=10, max=500),
...     'max_depth': IntHyperParam(min=3, max=20)
... }
>>> rf_tunable = Tunable(rf_hyperparams)
>>> svc_hyperparams = {
...     'C': FloatHyperParam(min=0.01, max=10.0),
...     'gamma': FloatHyperParam(0.000000001, 0.0000001)
... }
>>> svc_tunable = Tunable(svc_hyperparams)
>>> tuners = {
...     'RF': GPTuner(rf_tunable),
...     'SVC': GPTuner(svc_tunable)
... }
```

Then we perform the following steps in a loop.

1. Pass all the obtained scores to the selector and let it decide which model to test.

    ``` python
    >>> next_choice = selector.select({
    ...     'RF': tuners['RF'].scores,
    ...     'SVC': tuners['SVC'].scores
    ... })
    >>> next_choice
    'RF'
    ```

2. Obtain a new set of parameters from the indicated tuner and create a model instance.

    ``` python
    >>> parameters = tuners[next_choice].propose()
    >>> parameters
    {'n_estimators': 289, 'max_depth': 18}
    >>> model = models[next_choice](**parameters)
    ```

3. Evaluate the score of the new model instance and pass it back to the tuner

    ``` python
    >>> model.fit(X_train, y_train)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=18, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=289, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    >>> score = model.score(X_test, y_test)
    >>> score
    0.89
    >>> tuners[next_choice].record(parameters, score)
    ```

## What's next?

For more details about **BTB** and all its possibilities and features, please check the
[project documentation site](https://HDI-Project.github.io/BTB/)!

## Citing BTB

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
