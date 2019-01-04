<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“BTB” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

![](https://raw.githubusercontent.com/HDI-Project/BTB/master/docs/_static/BTB-Icon-small.png)

A simple, extensible backend for developing auto-tuning systems.

[![PyPI Shield](https://img.shields.io/pypi/v/baytune.svg)](https://pypi.python.org/pypi/baytune)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/BTB.svg?branch=master)](https://travis-ci.org/HDI-Project/BTB)

# Overview

Bayesian Tuning and Bandits is a simple, extensible backend for developing auto-tuning systems such as AutoML systems. It is currently being used in [ATM](https://github.com/HDI-Project/ATM) (an AutoML system that allows tuning of classifiers) and MIT's system for the DARPA [Data driven discovery of models program](https://www.darpa.mil/program/data-driven-discovery-of-models). 

* Free software: MIT license
* Documentation: https://HDI-Project.github.io/BTB
* Homepage: https://github.com/HDI-Project/BTB

*BTB is under active development. If you come across any issues, please report them [here](https://github.com/HDI-Project/BTB/issues/new).*

## Installation

### Install with pip

The easiest way to install BTB is using `pip`.

```
pip install baytune
```

### Build from source

You can also clone the repository and build it from source.

```
git clone git@github.com:HDI-Project/BTB.git
cd BTB
make install
```

## Basic Usage

### Tuners

In order to use a tuner we will create a ``Tuner`` instance indicating which parameters
we want to tune, their types and the range of values that we want to try.

``` python
>>> from btb.tuning import GP
>>> from btb import HyperParameter, ParamTypes
>>> tunables = [
... ('n_estimators', HyperParameter(ParamTypes.INT, [10, 500])),
... ('max_depth', HyperParameter(ParamTypes.INT, [3, 20]))
... ]
>>> tuner = GP(tunables)
```

Then we perform the following three steps in a loop.

1. Let the Tuner propose a new set of parameters

    ``` python
    >>> parameters = tuner.propose()
    >>> parameters
    {'n_estimators': 297, 'max_depth': 3}
    ```

2. Fit and score a new model using these parameters

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

3. Pass the used parameters and the score obtained back to the tuner

    ``` python
    tuner.add(parameters, score)
    ```

At each iteration, the `Tuner` will use the information about the previous tests
to evaluate and propose the set of parameter values that have the highest probability
of obtaining the highest score.

For more detailed examples, check scripts from the `examples` folder.

### Selectors

The selectors are intended to be used in combination with tuners in order to find
out and decide which model seems to get the best results once it is properly fine tuned.

In order to use the selector we will create a ``Tuner`` instance for each model that
we want to try out, as well as the ``Selector`` instance.

```
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.svm import SVC
>>> models = {
...     'RF': RandomForestClassifier,
...     'SVC': SVC
... }
>>> from btb.selection import UCB1
>>> selector = UCB1(['RF', 'SVC'])
>>> tuners = {
...     'RF': GP([
...         ('n_estimators', HyperParameter(ParamTypes.INT, [10, 500])),
...         ('max_depth', HyperParameter(ParamTypes.INT, [3, 20]))
...     ]),
...     'SVC': GP([
...         ('c', HyperParameter(ParamTypes.FLOAT_EXP, [0.01, 10.0])),
...         ('gamma', HyperParameter(ParamTypes.FLOAT, [0.000000001, 0.0000001]))
...     ])
... }
```

Then we perform the following steps in a loop.

1. Pass all the obtained scores to the selector and let it decide which model to test.

    ``` python
    >>> next_choice = selector.select({'RF': tuners['RF'].y, 'SVC': tuners['SVC'].y})
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
    >>> tuners[next_choice].add(parameters, score)
    ```

## References

If you use BTB, please consider citing the following work:

- Laura Gustafson. Bayesian Tuning and Bandits: An Extensible, Open Source Library for AutoML. Masters thesis, MIT EECS, June 2018. [(pdf)](https://dai.lids.mit.edu/wp-content/uploads/2018/05/Laura_MEng_Final.pdf)

  ``` bibtex 
  @MastersThesis{Laura:2018,
    title = "Bayesian Tuning and Bandits: An Extensible, Open Source Library for AutoML",
    author = "Laura Gustafson",
    month = "May",
    year = "2018",
    url = "https://dai.lids.mit.edu/wp-content/uploads/2018/05/Laura_MEng_Final.pdf",
    type = "M. Eng Thesis",
    address = "Cambridge, MA",
    school = "Massachusetts Institute of Technology",
  }
```
