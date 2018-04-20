[![][pypi-img]][pypi-url]
[![][travis-img]][travis-url]

# BTB: Bayesian Tuning and Bandits

Smart selection of hyperparameters

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/BTB

[travis-img]: https://travis-ci.org/HDI-Project/BTB.svg?branch=master
[travis-url]: https://travis-ci.org/HDI-Project/BTB
[pypi-img]: https://img.shields.io/pypi/v/btb.svg
[pypi-url]: https://pypi.python.org/pypi/btb

* selection/ defines Selectors: classes for choosing from a set of discrete
  options with multi-armed bandits
* tuning/ defines Tuners: classes with a fit/predict/propose interface for
  suggesting sets of hyperparameters
