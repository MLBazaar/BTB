#Tuning example using Rosenbrock function from https://en.wikipedia.org/wiki/Rosenbrock_function
f

def rosenbrock(x,y):
    a = 1
    b = 100
    return (a-x)**2 + b*(y-x**2)**2



#create hyperparameters x,y
hyperparams = [('x', HyperParameter(type='INT', range=(-100, 100))),
    ('y', HyperParameter(type='INT', range=(-100, 100)))]

tuner
