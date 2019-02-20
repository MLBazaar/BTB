"""
In this example, we use a Tuner to find the minimum of the Rosenbrock
function https://en.wikipedia.org/wiki/Rosenbrock_function

We compare the results given by a Uniform tuner and a GP-based tuner run for
100 iterations each.
"""

import btb.tuning
from btb import HyperParameter


def rosenbrock(x, y, a=1, b=100):
    """Bigger is better; global optimum at x=a, y=a**2"""
    return -1 * ((a - x)**2 + b * (y - x**2)**2)


def find_min_with_tuner(tuner, iterations=100):

    # main tuning loop
    for i in range(iterations):

        # use tuner to get next set of (x,y) to try
        candidate = tuner.propose()

        # score the candidate point (x, y) -- always doing maximization!
        score = rosenbrock(**candidate)

        # report the results back to the tuner
        tuner.add(candidate, score)

    print('best score: ', tuner._best_score)
    print('best hyperparameters: ', tuner._best_hyperparams)


# initialize the tunables, ie the function inputs x and y
# we make a prior guess that the maximum function value will be found when
# x and y are between -100 and 1000

tunables = (
    ('x', HyperParameter('float', [-100, 100])),
    ('y', HyperParameter('float', [-100, 100])),
)

print('Tuning with Uniform tuner')
tuner = btb.tuning.Uniform(tunables)
find_min_with_tuner(tuner)
print()

print('Tuning with GP tuner')
tuner = btb.tuning.GP(tunables)
find_min_with_tuner(tuner)
print()

actual = rosenbrock(1, 1)
print('Actual optimum: ', actual)
