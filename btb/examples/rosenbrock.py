from btb import HyperParameter, ParamTypes
from btb.tuning import Uniform, GP
import numpy as np

"""
In this example, we use a Tuner to estimate the minimum of a Rosenbrok
function https://en.wikipedia.org/wiki/Rosenbrock_function

We compare the results given by a Uniform tuner and a GP-based tuner run for
100 iterations each.
"""

def rosenbrok(x,y):
    a=1
    b=100
    return (a-x)**2 + b*(y-x**2)**2

def find_min_with_tuner(tuner):
    x_y = []
    rosenbrok_scores = []
    minimum_score = float("inf")
    xy_min = None
    for i in range(100):
        #use tuner to get next set of (x,y) to try
        xy_to_try = tuner.propose(np.array(x_y), np.array(rosenbrok_scores))
        x = xy_to_try[0]
        y = xy_to_try[1]
        score = rosenbrok(x,y)
        if score < minimum_score:
            minimum_score = score
            xy_min = (x,y)
        rosenbrok_scores.append(-1*score) #tuner needs a maximization problem
        x_y.append([x, y])

    print("minimum score:", minimum_score)
    print("minium score x:", xy_min[0], "minimum score y:", xy_min[1])

#initialize the tuneables, ie the function inputs x and y
#we make a prior guess that the mimum function value will be found when
#x and y are between -100 and 1000

x = HyperParameter('int', [-100, 1000])
y = HyperParameter('int', [-100, 1000])

print("------------Minimum found with uniform tuner--------------")
tuner = Uniform([("x",x), ("y", y)])
find_min_with_tuner(tuner)

print("------------Minimum found with GP tuner--------------")
tuner = GP([("x",x), ("y", y)])
find_min_with_tuner(tuner)
