# Benchmarking BTB

**BTB** provides a benchmarking framework that allows users and developers to evaluate the
performance of the BTB Tuners for Machine Learning Hyperparameter Tuning on hundreds of real world
classification problem and classical mathematical optimization problems.

As part of our benchmarking efforts we run the framework at every release and make the results
public. In each run we compare it to other tuners and optimizer libraries. We are constantly adding
new libraries for comparison. If you have suggestions for a tuner library we should include in our
compraison, please contact us via email at [dailabmit@gmail.com](mailto:dailabmit@gmail.com).

## The Benchmarking process

The Benchmarking BTB process has two main concepts.

### Challenges

A Challenge of the BTB Benchmarking framework is a Python class which has a method that produces a
score that can be optimized by tuning a set of hyperparameters.

There are two types of challenges: Machine Learning and Mathematical Optimization.

#### Machine Learning Challenges

The goal of these challenges is to optimize the score of Machine Learning models
over real world classification problems by tuning the hyperparameters of the model.

The following machine models are used:

- [Gradient Boosting](https://xgboost.readthedocs.io/en/latest/)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)

And the collection of datasets can be found in the [atm-data bucket on S3](http://atm-data.s3.amazonaws.com/index.html).

#### Mathematical Optimization Challenges

These challenges consist of classical Mathematical Optimization functions which the tuner attempts
to maximize by tuning its input parameters.

The following functions have been implemented:

- [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function).
- [Branin](https://www.sfu.ca/~ssurjano/branin.html).
- [Bohachevsky](https://www.sfu.ca/~ssurjano/boha.html).

### Tuning Functions

In the context of the BTB Benchmarking, `Tuning Functions` are python functions that, given a scoring
function and its tunable hyperparameters, try to search for the ideal hyperparameter values within
a given number of iterations.

In many cases tuning functions are based on the BTB Tuner classes, but some other functions have
been implemented using third party libraries to be able to compare their performance.

#### Tuning functions under comparison

Currently, benchmarking framework compares the following tuning functions from BTB:

- [BTB.Uniform](https://github.com/MLBazaar/BTB/blob/master/btb/tuning/tuners/uniform.py): Uses a Tuner that samples proposals randomly using a uniform distribution.
- [BTB.GPTuner](https://github.com/MLBazaar/BTB/blob/master/btb/tuning/tuners/gaussian_process.py): Uses a Bayesian Tuner that optimizes proposals using a GaussianProcess metamodel.
- [BTB.GPEiTuner](https://github.com/MLBazaar/BTB/blob/master/btb/tuning/tuners/gaussian_process.py): Uses a Bayesian Tuner that optimizes proposals using a GaussianProcess metamodel and an Expected Improvement acquisition function.
- [BTB.GCPTuner](https://github.com/MLBazaar/BTB/blob/master/btb/tuning/tuners/gaussian_process.py): Uses a Bayesian Tuner that optimizes proposals using a GaussianCopulaProcess metamodel.
- [BTB.GCPEiTuner](https://github.com/MLBazaar/BTB/blob/master/btb/tuning/tuners/gaussian_process.py): Uses a Bayesian Tuner that optimizes proposals using a GaussianCopulaProcess metamodel and an Expected Improvement acquisition function.

And the following external tuning functions:

- [HyperOpt.tpe](https://github.com/hyperopt/hyperopt/blob/master/hyperopt/tpe.py): Implements a Tree-Structured Parzen Estimator for hyperparameter search.
- [Ax.optimize](https://github.com/facebook/Ax): Implements Bayesian optimization and bandit optimization, powered by [BoTorch](https://github.com/pytorch/botorch).
- [SMAC.SMAC4HPO](https://github.com/automl/SMAC3/blob/master/smac/facade/smac_hpo_facade.py): Bayesian optimization using a Random Forest model of *pyrfr*.
- [SMAC.HB4AC](https://github.com/automl/SMAC3/blob/master/smac/facade/hyperband_facade.py): Uses Successive Halving for proposals.

Note: In our future releases we will be adding the following:

- [Sherpa](https://github.com/sherpa-ai/sherpa/).
- [GPyOpt](https://github.com/SheffieldML/GPyOpt).

#### To introduce a new Tuning Function:

If you want to add a tuner, you could follow the specific signature a tuning function has:

```python3
def tuning_function(
    scoring_function: callable,
    tunable_hyperparameters: dict,
    iterations: int) -> score: float
```

Please see how we introduced `HyperOpt` with this [signature here](https://github.com/MLBazaar/BTB/blob/master/benchmark/btb_benchmark/tuners/hyperopt.py).

## Running the Benchmarking

### Install

Before running the benchmarking process, you will have to follow this two steps in order to
install the package:

#### System Requierements

`BTB` benchmark has a system requierement of `swig (>=3.0,<4.0)` as build dependency. To install
it on a Ubuntu based machine you can run the following command:

```bash
sudo apt-get install swig
```

#### Python installation

You will have to install `BTB` from sources for development in order to use the benchmarking
package. To do so, clone the repository and run `make install-develop`:

```bash
git clone git@github.com:MLBazaar/BTB.git
cd BTB
make install-develop
```

### Runnig the Benchmarking using python

The user API for the BTB Benchmarking is the `btb_benchmark.main.run_benchmark` function.

The simplest usage is to execute the `run_benchmark` function without any arguments:

```python
from btb_benchmark import run_benchmark

scores = run_benchmark()
```

This will execute all the tuners that have been implemented in `btb_benchmark` on all the
challenges and return a `pandas.DataFrame` with one column per Challenge and one row per Tuner
containing the best scores obtained by each combination:

```
                                                       BTB.GPEiTuner  ...  HyperOpt.rand.suggest  HyperOpt.tpe.suggest
0              XGBoostChallenge('PizzaCutter1_1.csv')       0.664602  ...               0.617456              0.658357
1        XGBoostChallenge('analcatdata_apnea3_1.csv')       0.812482  ...               0.800539              0.821088
2                       XGBoostChallenge('ar4_1.csv')       0.675651  ...               0.622220              0.633233
3                       XGBoostChallenge('ar5_1.csv')       0.547937  ...               0.445195              0.445195
4                     XGBoostChallenge('ecoli_1.csv')       0.728677  ...               0.705936              0.724864
5             XGBoostChallenge('eye_movements_1.csv')       0.834001  ...               0.820084              0.824663
```

### Benchmark Arguments

The `run_benchmark` function has the following arguments:

- `tuners`: list of tuners that will be benchmarked.
- `challenge_types`: list of types of challenges that will be used for benchmark (optional).
- `challenges`: list of names of challenges that will be benchmarked (optional).
- `sample`: if specified, run the benchmark on a subset of the available challenges of the given size (optional).
- `iterations`: the number of tuning iterations to perform per challenge and tuner.
- `max_rows`: Number of rows from the dataframe to be used (MLChallenges only). Defaults to `None`.
- `output_path`: If given, store the benchmark results in the given path as a CSV file.

#### Tuners

If you want to run the benchmark on your own tuner implementation, or in a subset of the BTB
tuners, you can pass them as a list to the tuners argument. This can be done by either directly
passing the function or the name of the tuner.

For example, if we want to compare the performance of our tuning function and BTB.GPTuner, we can
call the `run_benchmark` function like this:

```python3
tuners = [
    my_tuning_function,
    'BTB.GPTuner',
]
results = run_benchmark(tuners=tuners)
```

#### Challenges

If we want to run the benchmark on a subset of the challenges, we can pass their names to the
challenges argument. If a given challenge is the name of a mathematical optimization problem
function, the corresponding Mathematical Optimization Challenge will be executed.

If the given challenge is the name of a Machine Learning Classification problem, all the
implemented classifiers will be benchmarked on that dataset.

For example, if we want to run only on the Rosenbrock function and evaluate on the `stock_1`
dataset, we can call the `run_benchmark` function like this:

```python3
challenges = ['rosenbrock', 'stock_1']
results = run_benchmark(challenges=challenges)
```

Additionally, if we only want to run on a family of challenges or a specific Machine Learning
model, we can specify it passing the `types` argument.

For example, if we want to run all the dataset on the XGBoost model, we can call the run benchmark
function like this:

```python3
results = run_benchmark(challenge_types=['xgboost'])
```

Finally, if we want to further reduce the amount of challenges that are executed, we can run on a
random subsample of all the selected challenges using the `sample` argument.

For example, if we want to run `XGBoost` only on 10 random datasets, we can use:

```python3
results = run_benchmark(challenge_types=['xgboost'], sample=10)
```

#### Iterations

By default the benchmark runs 100 tuning iterations per tuner and challenge. If we want to change
the amount of iterations to be performed by each tuner, we can specify so by adding the argument
`iterations`.

For example, if we want to run all the challenges for 1000 iterations / tuner each we can use:

```python3
results = run_benchmark(iterations=1000)
```

#### Storing the results

If we want to store the obtained results directly in to a file, we can pass the path to where we
would like to save our results, by adding the argument `output_path`.

For example, if we want to store it as `path/to/my_results.csv` we can use:

```python3
run_benchmark(output_path='path/to/my_results.csv')
```

## Our latest Results

All the results obtained by the different BTB releases can be found inside the
[results](https://github.com/MLBazaar/BTB/tree/master/benchmark/results) folder as CSV files.

Additionally, all the previous results can be browsed and analyzed in the following [Google Sheets
document](
https://docs.google.com/spreadsheets/d/15a-pAV_t7CCDvqDyloYmdVNFhiKJFOJ7bbgpmYIpyTs/edit?usp=sharing).


## Kubernetes

Running the complete BTB Benchmarking suite can take a long time when executing against all our
challenges. For this reason, it comes prepared to be executed distributedly over a dask cluster
created using Kubernetes. Check our [documentation](https://mlbazaar.github.io/BTB/kubernetes.html)
on how to run on a kubernetes cluster.


## Credits

All the datasets used for the BTB benchmarking were downloaded from [openml.org](openml.org).

Full details about their origin can be read the paper by Joaquin Vanschoren, Jan N. van Rijn,
Bernd Bischl, and Luis Torgo. [OpenML: networked science in machine learning](
http://arxiv.org/abs/1407.7722). SIGKDD Explorations 15(2), pp 49-60, 2013.

After the download, the datasets went through a cleanup process in which the following
modifications were applied:
* Encode categorical columns with numerical values.
* Rename the target column to `class` and make it the last column.
