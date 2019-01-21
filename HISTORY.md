# History

## 0.2.2

### Internal Improvements

* Updated documentation

### Bug fixes

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

### Bug fixes

* Fix error from mixing string/numerical hyperparameters
* Inverse transform for categorical hyperparameter returns single item

## 0.1.2

* Issue #47: Add missing requirements in v0.1.1 setup.py
* Issue #46: Error on v0.1.1: 'GP' object has no attribute 'X'

## 0.1.1

* First release.
