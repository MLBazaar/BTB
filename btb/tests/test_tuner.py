import numpy as np
from btb import ParamTypes, HyperParameter
from btb.tuning import Uniform, GP, GPEi, GPEiVelocity, GCP, GCPEi, GCPEiVelocity
import unittest

class TestHyperparameter(unittest.TestCase):
    def test_uniform(self):
        X = np.array([[1.1,0.01,3.5, 'a'],[4,0.001,6.2, 'b']])
        y = np.array([0.5, 0.6])
        c1 = HyperParameter(ParamTypes.INT, [1,5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001,0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2,8])
        c4 = HyperParameter(ParamTypes.STRING, ['a', 'b', 'c'])
        tunables = [('t1', c1),('t2', c2),('t3',c3), ('t4', c4)]
        u = Uniform(tunables)
        for i in range(100):
            proposed = u.propose(X, y)
            self.assertTrue(proposed[0] >=1 and proposed[0] <=5)
            self.assertTrue(proposed[1] >=0.0001 and proposed[1] <=0.1)
            self.assertTrue(proposed[2] >=2 and proposed[2] <=8)
            self.assertTrue(proposed[3] in ['a', 'b', 'c'])
            
    def test_gp(self):
        X = np.array([[1.1,0.01,3.5],[4,0.001,6.2]])
        y = np.array([0.5, 0.6])
        c1 = HyperParameter(ParamTypes.INT, [1,5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001,0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2,8])
        tunables = [('a', c1),('b', c2),('c',c3)]
        u = GP(tunables)
        for i in range(100):
            proposed = u.propose(X, y)
            self.assertTrue(proposed[0] >=1 and proposed[0] <=5)
            self.assertTrue(proposed[1] >=0.0001 and proposed[1] <=0.1)
            self.assertTrue(proposed[2] >=2 and proposed[2] <=8)

    def test_gpei(self):
        X = np.array([[1.1,0.01,3.5],[4,0.001,6.2]])
        y = np.array([0.5, 0.6])
        c1 = HyperParameter(ParamTypes.INT, [1,5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001,0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2,8])
        tunables = [('a', c1),('b', c2),('c',c3)]
        u = GPEi(tunables)
        for i in range(100):
            proposed = u.propose(X, y)
            self.assertTrue(proposed[0] >=1 and proposed[0] <=5)
            self.assertTrue(proposed[1] >=0.0001 and proposed[1] <=0.1)
            self.assertTrue(proposed[2] >=2 and proposed[2] <=8)

    def test_gpeivelocity(self):
        X = np.array([[1.1,0.01,3.5],[4,0.001,6.2]])
        y = np.array([0.5, 0.6])
        c1 = HyperParameter(ParamTypes.INT, [1,5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001,0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2,8])
        tunables = [('a', c1),('b', c2),('c',c3)]
        u = GPEiVelocity(tunables)
        for i in range(100):
            proposed = u.propose(X, y)
            self.assertTrue(proposed[0] >=1 and proposed[0] <=5)
            self.assertTrue(proposed[1] >=0.0001 and proposed[1] <=0.1)
            self.assertTrue(proposed[2] >=2 and proposed[2] <=8)

    def test_gcp(self):
        X = np.array([[1.1,0.01,3.5],[4,0.001,6.2]])
        y = np.array([0.5, 0.6])
        c1 = HyperParameter(ParamTypes.INT, [1,5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001,0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2,8])
        tunables = [('a', c1),('b', c2),('c',c3)]
        u = GCP(tunables)
        proposed = u.propose(X, y)
        self.assertTrue(proposed[0] >=1 and proposed[0] <=5)
        self.assertTrue(proposed[1] >=0.0001 and proposed[1] <=0.1)
        self.assertTrue(proposed[2] >=2 and proposed[2] <=8)

    def test_gcpei(self):
        X = np.array([[1.1,0.01,3.5],[4,0.001,6.2]])
        y = np.array([0.5, 0.6])
        c1 = HyperParameter(ParamTypes.INT, [1,5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001,0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2,8])
        tunables = [('a', c1),('b', c2),('c',c3)]
        u = GCPEi(tunables)
        proposed = u.propose(X, y)
        self.assertTrue(proposed[0] >=1 and proposed[0] <=5)
        self.assertTrue(proposed[1] >=0.0001 and proposed[1] <=0.1)
        self.assertTrue(proposed[2] >=2 and proposed[2] <=8)
    def test_gcpeivelocity(self):
        X = np.array([[1.1,0.01,3.5],[4,0.001,6.2]])
        y = np.array([0.5, 0.6])
        c1 = HyperParameter(ParamTypes.INT, [1,5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001,0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2,8])
        tunables = [('a', c1),('b', c2),('c',c3)]
        u = GCPEiVelocity(tunables)
        proposed = u.propose(X, y)
        self.assertTrue(proposed[0] >=1 and proposed[0] <=5)
        self.assertTrue(proposed[1] >=0.0001 and proposed[1] <=0.1)
        self.assertTrue(proposed[2] >=2 and proposed[2] <=8)
if __name__ == '__main__':
    unittest.main()
