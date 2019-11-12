# History

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
