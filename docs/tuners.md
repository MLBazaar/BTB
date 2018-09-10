# Tuners

Tuners are specifically designed to speed up the process of selecting the
optimal hyper parameter values for a specific machine learning algorithm.

`btb.tuning` defines Tuners: classes with a fit/predict/propose interface for suggesting sets of hyperparameters.

This is done by following a Bayesian Optimization approach and iteratively:

* letting the tuner propose new sets of hyper parameter
* fitting and scoring the model with the proposed hyper parameters
* passing the score obtained back to the tuner

At each iteration the tuner will use the information already obtained to propose
the set of hyper parameters that it considers that have the highest probability
to obtain the best results.

