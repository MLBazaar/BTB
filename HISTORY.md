# History

## 0.3.4 - 2019-12-24

With this release we introduce a `BTBSession` class. This class represents the process of selecting
and tuning several tunables until the best possible configuration fo a specific `scorer` is found.
We also have improved and fixed some minor bugs arround the code (described in the issues below).

### New Features

* `BTBSession` that makes `BTB` more user friendly.

### Internal Improvements

Improved unittests, removed old dependencies, added more `MLChallenges` and fixed an issue with
the bound methods.

### Resolved Issues

* Issue #145: Implement `BTBSession`.
* Issue #155: Set defaut to `None` for `CategoricalHyperParam` is not possible.
* Issue #157: Metamodel `_MODEL_KWARGS_DEFAULT` becomes mutable.
* Issue #158: Remove `mock` dependency from the package.
* Issue #160: Add more Machine Learning Challenges and more estimators.


## 0.3.3 - 2019-12-11

Fix a bug where creating an instance of `Tuner` ends in an error.

### Internal Improvements

Improve unittests to use `spec_set` in order to detect errors while mocking an object.

### Resolved Issues

* Issue #153: Bug with tunner logger message that avoids creating the Tunner.

## 0.3.2 - 2019-12-10

With this release we add the new `benchmark` challenge `MLChallenge` which allows users to
perform benchmarking over datasets with machine learning estimators, and also some new
features to make the workflow easier.

### New Features

* New `MLChallenge` challenge that allows performing crossvalidation over datasets and machine
learning estimators.
* New `from_dict` function for `Tunable` class in order to instantiate from a dictionary that
contains information over hyperparameters.
* New `default` value for each hyperparameter type.

### Resolved Issues

* Issue #68: Remove `btb.tuning.constants` module.
* Issue #120: Tuner repr not helpful.
* Issue #121: HyperParameter repr not helpful.
* Issue #141: Imlement propper logging to the tuning section.
* Issue #150: Implement Tunable `from_dict`.
* Issue #151: Add default value for hyperparameters.
* Issue #152: Support `None` as a choice in `CategoricalHyperPrameters`.

## 0.3.1 - 2019-11-25

With this release we introduce a `benchmark` module for `BTB` which allows the users to perform
a benchmark over a series of `challenges`.

### New Features

* New `benchmark` module.
* New submodule named `challenges` to work toghether with `benchmark` module.

### Resolved Issues

* Issue #139: Implement a Benchmark for BTB

## 0.3.0 - 2019-11-11

With this release we introduce an improved `BTB` that has a major reorganization of the project
with emphasis on an easier way of interacting with `BTB` and an easy way of developing, testing and
contributing new acquisition functions, metamodels, tuners  and hyperparameters.

### New project structure

The new major reorganization comes with the `btb.tuning` module. This module provides everything
needed for the `tuning` process and comes with three new additions `Acquisition`, `Metamodel` and
`Tunable`. Also there is an update to the `Hyperparamters` and `Tuners`. This changes are meant
to help developers and contributors to easily develop, test and contribute new `Tuners`.

### New API

There is a slightly new way of using `BTB` as the new `Tunable` class is introduced, that is meant
to be the only requiered object to instantiate a `Tuner`. This `Tunable` class represents a
collection of `HyperParams` that need to be tuned as a whole, at once. Now, in order to create a
`Tuner`, a `Tunable` instance must be created first with the `hyperparameters` of the
`objective function`.

### New Features

* New `Hyperparameters` that allow an easier interaction for the final user.
* New `Tunable` class that manages a collection of `Hyperparameters`.
* New `Tuner` class that is a python mixin that requieres of `Acquisition` and `Metamodel` as
parents. Also now works with a single `Tunable` object.
* New `Acquisition` class, meant to implement an acquisition function to be inherit by a `Tuner`.
* New `Metamodel` class, meant to implement everything that a certain `model` needs and be inherit
by the `Tuner`.
* Reorganization of the `selection` module to follow a similar `API` to `tuning`.

### Resolved Issues

* Issue #131: Reorganize the project structure.
* Issue #133: Implement Tunable class to control a list of hyperparameters.
* Issue #134: Implementation of Tuners for the new structure.
* Issue #140: Reorganize selectors.

## 0.2.5

### Bug Fixes

* Issue #115: HyperParameter subclass instantiation not working properly

## 0.2.4

### Internal Improvements

* Issue #62: Test for `None` in `HyperParameter.cast` instead of `HyperParameter.__init__`

### Bug fixes

* Issue #98: Categorical hyperparameters do not support `None` as input
* Issue #89: Fix the computation of `avg_rewards` in `BestKReward`

## 0.2.3

### Bug Fixes

* Issue #84: Error in GP tuning when only one parameter is present bug
* Issue #96: Fix pickling of HyperParameters
* Issue #98: Fix implementation of the GPEi tuner

## 0.2.2

### Internal Improvements

* Updated documentation

### Bug Fixes

* Issue #94: Fix unicode `param_type` caused error on python 2.

## 0.2.1

### Bug fixes

* Issue #74: `ParamTypes.STRING` tunables do not work

## 0.2.0

### New Features

* New Recommendation module
* New HyperParameter types
* Improved documentation and examples
* Fully tested Python 2.7, 3.4, 3.5 and 3.6 compatibility
* HyperParameter copy and deepcopy support
* Replace print statements with logging

### Internal Improvements

* Integrated with Travis-CI
* Exhaustive unit testing
* New implementation of HyperParameter
* Tuner builds a grid of real values instead of indices
* Resolve Issue #29: Make args explicit in `__init__` methods
* Resolve Issue #34: make all imports explicit

### Bug Fixes

* Fix error from mixing string/numerical hyperparameters
* Inverse transform for categorical hyperparameter returns single item

## 0.1.2

* Issue #47: Add missing requirements in v0.1.1 setup.py
* Issue #46: Error on v0.1.1: 'GP' object has no attribute 'X'

## 0.1.1

* First release.
