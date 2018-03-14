import numpy as np
from btb import ParamTypes, HyperParameter
from btb.tuning import Uniform
import unittest

class TestHyperparameter(unittest.TestCase):
    def test_uniform(self):
        X = np.array([[1.1,2.2,3.5],[4,5.1,6.2]])
        y = np.array([0.5, 0.6])
        c1 = HyperParameter(ParamTypes.INT, [1,5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [1,7])
        c3 = HyperParameter(ParamTypes.FLOAT, [2,8])
        tunables = [('a', c1),('b', c2),('c',c3)]
        u = Uniform(tunables)
        for i in range(100):
            proposed = u.propose(X, y)
            self.assertTrue(proposed[0] >=1 and proposed[0] <=5)
            self.assertTrue(proposed[1] >=1 and proposed[1] <=7)
            self.assertTrue(proposed[2] >=2 and proposed[2] <=8)

if __name__ == '__main__':
    unittest.main()
