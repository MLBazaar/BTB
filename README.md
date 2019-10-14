<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“BTB” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPi Shield](https://img.shields.io/pypi/v/baytune.svg)](https://pypi.python.org/pypi/baytune)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/BTB.svg?branch=master)](https://travis-ci.org/HDI-Project/BTB)
[![Coverage Status](https://codecov.io/gh/HDI-Project/BTB/branch/master/graph/badge.svg)](https://codecov.io/gh/HDI-Project/BTB)
[![Downloads](https://pepy.tech/badge/baytune)](https://pepy.tech/project/baytune)

<br />

![](https://raw.githubusercontent.com/HDI-Project/BTB/master/docs/_static/BTB-Icon-small.png)

<br />

- License: MIT
- Documentation: https://hdi-project.github.io/BTB
- Homepage: https://github.com/hdi-project/BTB

# Overview

**BTB** is a Python library that allows a simple, extensible backend for developing auto-tuning
systems such as AutoML systems.

**BTB** can be used in any `objective function` which has as an input `n` number of parameters
that can change their value in order to obtain a different score. In machine learning, this also
applies to the `hyperparameters` that the models are used during instantiation.

# Concepts

Before diving into the software usage, we briefly explain some concepts and terminology.

## Hyperparameters

Hyperparameters  are variables that affect an `objective function`. We can find the following
types of hyperparameters in **BTB**:

- Numerical Hyperparameters: These hyperparameters consist of a single value which has to be a
number within a defined range.
- Boolean Hyperparameters: These hyperparameters consist of a single value which can be either
`True` or `False`.
- Categorical Hyperparameters: These hyperparameters consist of a single value which has to be
taken from a list of valid choices.

## Tunable

Tunable is designed to work with a collection of hyperparameters that need to be tuned as a
whole, at once.

## Tuners

Tuners are specifically designed to speed up the process of selecting the optimal `hyperparameter`
values for a specific `objective function`. This class works with a `tunable` object in order
to generate new proposals and learn from them, saving fitting time by predicting the score for
the proposed configurations and applying an `AcquisitionFunction` over those predictions.

## Selector

Selectors apply multiple strategies to decide which `tuner` train and test next based on how well
they have been performing in the previous test runs. This is an application of what is called
the Multi-armed Bandit Problem.

The process works by letting know the selector which `tuners` have been already tested
and which scores they have obtained, and letting it decide which `tuner` to test next.

# Install

## Requirements

**BTB** has been developed and tested on [Python 3.5, 3.6 and 3.7](https://www.python.org/downloads)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **BTB** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **BTB**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) btb-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source btb-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **BTB**!

## Install using Pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **BTB**:

```bash
pip install btb
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from Source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:HDI-Project/BTB.git
cd BTB
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://HDI-Project.github.io/BTB/contributing.html#get-started)
for more details about this process.


# Quickstart

In this short series of tutorials we will guide you through a series of steps that will
help you getting started using **BTB** for tuning your models or `functions`.

## Objective function

Here we will define a simple objective function to use thro this example, it's a well known
[rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).

```python
def rosenbrock(x, y, a=1, b=100):
    return -1 * ((a - x)**2 + b * (y - x**2)**2)
```

This function takes as input two `hyperparameters` which are `x` and `y`. This is a maximization
function where bigger is better and the global optimum is at `x=a` and `y=a**2`.

## Hyperparameters

For our objective function, we will create two `IntHyperParameter` with `min=-10` being this the
minimum value that we would like to take, and `max=10` being this the maximum value that we would
like to obtain during the proposals.

Creating a hyperparameter is a simple as importing it's class from `btb.tuning.hyperparams` and
call it with the desired arguments.

**Bear in mind**  that the `BooleanHyperParam` doesn't require any arguments, while
`CategoricalHyperParam` requieres of a `list` with the possible choices. The numerical
hyperparameters can take optionally `min` and `max` to establish their range. For more information
head over the [hyperparameter's api documentation site](https://HDI-Project.github.io/BTB/api/btb.tuning.hyperparams.html)!

```python
from btb.tuning.hyperparams import IntHyperParam

xihp = IntHyperParam(min=-10, max=10)
yihp = IntHyperParam(min=-10, max=10)
```

## Tunable

In order to create a `Tunable` object, first you have to import the class from `btb.tuning.tunable`
and then create a dictionary that contains the name of the variable and the instance of the
`hyperparameter`. In this case we will create a simple dictionary containing the two instances
that we created before, and the `x` and `y` as keys.

```python
from btb.tuning.tunable import Tunable

hyperparameters = {
    'x': xihp,
    'y': yihp
}

tunable = Tunable(hyperparameters)
```

## Tuners

Once we have our `tunable` we it's time to create a tuner in order to be able to work with this
object. For this example we will use `UniformTuner`. To import it, simply import it from
`btb.tuning.tuners`:

```python
from btb.tuning.tuners import UniformTuner

tuner = UniformTuner(tunable)
```

## Propose and record

Once we have our `tuner` we can start generating proposals by simply calling it's method `propose`:

```python
proposed = tuner.propose()
```

If we check what's inside `proposed` we will see that some random values between `-10` and `10`
where generated for `x` and `y` and are returned as a `dict` in order to be able to use them
as `kwargs`.

```python
proposed

{'x': 2, 'y': 5}
```

Now we can use this dictionary to generate a `score` using our objective function:

```python
score = rosenbrock(**proposed)
```

After obtaining a `score` we can record the `proposed` parameters with the `score` for them.

```python
tuner.record(proposed, score)
```

This will record the trial, and unless we use `allow_duplicates=True` when calling the `propose`
method of the `tuner` only new configurations will be generated.

## Selectors

Selectors apply multiple strategies to decide which models or families of models to
train and test next based on how well thay have been performing in the previous test runs.
This is an application of what is called the Multi-armed Bandit Problem.

`btb.selection` defines Selectors: classes for choosing from a set of discrete options with multi-armed bandits.

The process works by letting know the selector which models have been already tested
and which scores they have obtained, and letting it decide which model to test next.

The selectors are intended to be used in combination with tuners in order to find
out and decide which model seems to get the best results once it is properly fine tuned.

In order to use the selector we will create a ``Tuner`` instance for each model that
we want to try out, as well as the ``Selector`` instance.

```python
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

## What's next?
For more details about **BTB** and all its possibilities and features, please check the
[project documentation site](https://HDI-Project.github.io/BTB/)!

## Citing BTB

If you use BTB, please consider citing the following work:

- Laura Gustafson. Bayesian Tuning and Bandits: An Extensible, Open Source Library for AutoML. Masters thesis, MIT EECS, June 2018. [(pdf)](https://dai.lids.mit.edu/wp-content/uploads/2018/05/Laura_MEng_Final.pdf)

BibTeX entry:

```bibtex
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

# Related Projects

* **ATM**: https://github.com/HDI-Project/ATM
* **AutoBazaar**: https://github.com/HDI-Project/AutoBazaar
* **mit-d3m-ta2**: https://github.com/HDI-Project/mit-d3m-ta2
