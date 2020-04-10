# BTB Benchmark
**BTB** provides a benchmark framework that allows users and developers to evaluate the performance
of the BTB Tuners on multiple optimization problems, including Machine Learning Hyperparameter
Tuning on hundreds of real world classification problems and classical mathematical optimization
problems.

## The Benchmarking process

The BTB Benchmark process spins around two main concepts

### Challenges

A Challenge of the BTB Benchmark framework is a Python class which has a method that produces a
score that can be optimized by tuning a set of hyperparameters.

There are two types of challenges: Machine Learning and Mathematical Optimization.

#### Machine Learning Challenges

The goal of these challenges is to optimize the goodness-of-fit score of Machine Learning models
over real world classification problems by tuning the hyperparameters of the model.

The following machine models have been implemented

- Gradient Boosting: https://xgboost.readthedocs.io/en/latest/
- Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Stochastic Gradient Descent: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier

And the collection of datasets can be found in the atm-data bucket on S3: http://atm-data.s3.amazonaws.com/index.html

#### Mathematical Optimization Challenges
These challenges consist of classical Mathematical Optimization functions which the tuner attempts
to maximize by tuning its input arguments.

The following functions have been implemented:

- Rosenbrock: https://en.wikipedia.org/wiki/Rosenbrock_function
- Branin: https://www.sfu.ca/~ssurjano/branin.html
- Bohachevsky: https://www.sfu.ca/~ssurjano/boha.html

### Tuners
In the context of the BTB Benchmark, Tuners are Python functions that, given a challenge scoring
function and its hyperparameters configuration, try to search for the ideal hyperparameter values
within a given number of iterations.

In many cases tuning functions are based on the BTB Tuner classes, but some other functions have
been implemented using third party libraries to be able to compare their performance.

The tuning functions have a very specific signature:

```python3
def tuning_function(
scoring_function: callable,
tunable_hyperparameters: dict,
iterations: int) -> score: float
```

#### Implemented Tuning Functions
Currently, BTB implements the following tuning functions:

- BTB.Uniform: Uses a Bayesian Tuner that samples proposals randomly using a uniform distribution.
- BTB.GPTuner: Uses a Bayesian Tuner that optimizes proposals using a GaussianProcess metamodel.
- BTB.GPEiTuner: Uses a Bayesian Tuner that optimizes proposals using a GaussianProcess metamodel
and an Expected Improvement acquisition function.
- HyperOpt.rand: Implements a random hyper parameter search.
- HyperOpt.tpe: Implements a Tree-Structured Parzen Estimator for hyperparameter search.

## Executing the Benchmark
The user API for the BTB Benchmark is the `btb_benchmark.main.run_benchmark` function.

The simplest usage is to execute the `run_benchmark` function without any arguments:

```python
from btb_benchmark import run_benchmark

df = run_benchmark()
```

This will execute all the tuners that have been implemented in `btb_benchmark` on all the
challenges and return a `pandas.DataFrame` with one column per Challenge and one row per Tuner
containing the best scores obtained by each combination:

|                                                    |BTB.GPEiTuner      |BTB.GPTuner        |BTB.UniformTuner   |HyperOpt.rand.suggest|HyperOpt.tpe.suggest|
|----------------------------------------------------|-------------------|-------------------|-------------------|---------------------|--------------------|
|XGBoostChallenge('BNG(cmc,nominal,55296)_1.csv')    |0.5199326904521441 |0.5201074137655625 |0.5203276907572528 |0.5191030441186218   |0.5199238818278141  |
|XGBoostChallenge('PieChart2_1.csv')                 |0.4945739104060386 |0.4945739104060386 |0.4945739104060386 |0.4945739104060386   |0.4945739104060386  |
|XGBoostChallenge('PizzaCutter1_1.csv')              |0.6646017083091733 |0.6551432921219396 |0.6273517348937345 |0.6174557518773459   |0.6583569881072966  |
|XGBoostChallenge('SPECT_1.csv')                     |0.7454525985019169 |0.7437053902170181 |0.7460349918100205 |0.7437053902170181   |0.7493155866991348  |
|XGBoostChallenge('aids_1.csv')                      |0.3333333333333333 |0.3333333333333333 |0.3333333333333333 |0.3333333333333333   |0.3333333333333333  |
|XGBoostChallenge('analcatdata_apnea3_1.csv')        |0.8124818879766957 |0.8107485926033646 |0.8116311831827117 |0.8005389942788316   |0.8210876768837629  |
|XGBoostChallenge('ar4_1.csv')                       |0.6756512903881324 |0.6672870902747065 |0.6151478457670407 |0.6222202422202423   |0.6332330827067668  |

### Benchmark Arguments

The `run_benchmark` function has the following arguments:

- tuners: Specify the tuners that will be benchmarked.
- types: Specify the families of challenges that will be used for the benchmark.
- challenges: Specify the challenges that will be benchmarked.
- sample: If specified, run the benchmark on a subset of the available challenges of the given size.
- iterations: the number of tuning iterations to perform per challenge and tuner.
- output_path: If given, store the benchmark results in the given path as a CSV file.

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
challenges = [‘rosenbrock’, ‘stock_1’]
results = run_benchmark(challenges=challenges)
```

Additionally, if we only want to run on a family of challenges or a specific Machine Learning
model, we can specify it passing the `types` argument.

For example, if we want to run all the dataset on the XGBoost model, we can call the run benchmark
function like this:

```python3
results = run_benchmark(types=['xgboost'])
```

Finally, if we want to further reduce the amount of challenges that are executed, we can run on a
random subsample of all the selected challenges using the `sample` argument.

For example, if we want to run `XGBoost` only on 10 random datasets, we can use:

```python3
results = run_benchmark(types=['xgboost'], sample=10)
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
run_benchmark(output_path=’path/to/my_results.csv’)
```

## Results
All the results obtained by the different BTB releases can be found inside the
[results](https://github.com/HDI-Project/BTB/tree/master/btb_benchmark/results) folder as CSV files.

Additionally, all the previous results can be browsed and analyzed in the following Google Sheets
document: https://drive.google.com/file/d/1yRwlCNien3EvbBtT6EwufuLfBHAempzr/view?usp=sharing.
