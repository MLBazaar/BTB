[![][pypi-img]][pypi-url]
[![][travis-img]][travis-url]

[travis-img]: https://travis-ci.org/HDI-Project/BTB.svg?branch=master
[travis-url]: https://travis-ci.org/HDI-Project/BTB
[pypi-img]: https://img.shields.io/pypi/v/baytune.svg
[pypi-url]: https://pypi.python.org/pypi/baytune

# BTB: Bayesian Tuning and Bandits

Smart selection of hyperparameters

* Free software: MIT license
* Documentation: https://HDI-Project.github.io/BTB

## Overview

Bayesian Tuning and Bandits is a simple, extensible Auto Machine Learning system
that automates model selection and hyperparameter tuning.

## Submodules

* `selection` defines Selectors: classes for choosing from a set of discrete
  options with multi-armed bandits
* `tuning` defines Tuners: classes with a fit/predict/propose interface for
  suggesting sets of hyperparameters

### Tuners

Tuners are specifically designed to speed up the process of selecting the
optimal hyper parameter values for a specific machine learning algorithm.

This is done by following a Bayesian Optimization approach and iteratively:

* letting the tuner propose new sets of hyper parameter
* fitting and scoring the model with the proposed hyper parameters
* passing the score obtained back to the tuner

At each iteration the tuner will use the information already obtained to propose
the set of hyper parameters that it considers that have the highest probability
to obtain the best results.

### Selectors

Selectors apply multiple strategies to decide which models or families of models to
train and test next based on how well thay have been performing in the previous test runs.
This is an application of what is called the Multi-armed Bandit Problem.

The process works by letting know the selector which models have been already tested
and which scores they have obtained, and letting it decide which model to test next.

## Installation

### Install with pip

The easiest way to install BTB is using `pip`

```
pip install baytune
```

### Install from sources

You can also clone the repository and install it from sources

```
git clone git@github.com:HDI-Project/BTB.git
cd BTB
make install
```

## Usage examples

### Tuners

In order to use a Tuner we will create a Tuner instance indicating which parameters
we want to tune, their types and the range of values that we want to try

```
>>> from btb.tuning import GP
>>> from btb import HyperParameter, ParamTypes
>>> tunables = [
... ('n_estimators', HyperParameter(ParamTypes.INT, [10, 500])),
... ('max_depth', HyperParameter(ParamTypes.INT, [3, 20]))
... ]
>>> tuner = GP(tunables)
```

Then we into a loop and perform three steps:

#### 1. Let the Tuner propose a new set of parameters

```
>>> parameters = tuner.propose()
>>> parameters
{'n_estimators': 297, 'max_depth': 3}
```

#### 2. Fit and score a new model using these parameters

```
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

#### 3. Pass the used parameters and the score obtained back to the tuner

```
tuner.add(parameters, score)
```

At each iteration, the Tuner will use the information about the previous tests
to evaluate and propose the set of parameter values that have the highest probability
of obtaining the highest score.

For a more detailed example, check scripts from the `examples` folder.

### Selectors

The selectors are intended to be used in combination with the Tuners in order to find
out and decide which model seems to get the best results once it is properly fine tuned.

In order to use the selector we will create a Tuner instance for each model that
we want to try out, as well as the selector instance.

```
>>> from sklearn.svm import SVC
>>> models = {
...     'RF': RandomForestClassifier,
...     'SVC': SVC
... }
>>> from btb.selection import UCB1
>>> selector = UCB1(['RF', 'SVM'])
>>> tuners = {
...     'RF': GP([
...         ('n_estimators', HyperParameter(ParamTypes.INT, [10, 500])),
...         ('max_depth', HyperParameter(ParamTypes.INT, [3, 20]))
...     ]),
...     'SVM': GP([
...         ('c', HyperParameter(ParamTypes.FLOAT_EXP, [0.01, 10.0])),
...         ('gamma', HyperParameter(ParamTypes.FLOAT, [0.000000001, 0.0000001]))
...     ])
... }
```

Then, we will go into a loop and, at each iteration, perform the steps:

#### 1. Pass all the obtained scores to the selector and let it decide which model to test

```
>>> next_choice = selector.select({'RF': tuners['RF'].y, 'SVM': tuners['SVM'].y})
>>> next_choice
'RF'
```

#### 2. Obtain a new set of parameters from the indicated tuner and create a model instance

```
>>> parameters = tuners[next_choice].propose()
>>> parameters
{'n_estimators': 289, 'max_depth': 18}
>>> model = models[next_choice](**parameters)
```

#### 3. Evaluate the score of the new model instance and pass it back to the tuner

```
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
>>> tuners[next_choice].add(parameters, score)
```
