from btb import ParamTypes, HyperParameter
from btb.tuning import Uniform, GP, GPEi, GPEiVelocity, GCP, GCPEi, GCPEiVelocity
import numpy as np
import unittest

class TestHyperparameter(unittest.TestCase):
    def test_uniform(self):
        X = [
            {'t1': 1.1, 't2': 0.01, 't3': 3.5, 't4': 'a'},
            {'t1': 4, 't2': 0.001, 't3': 6.2, 't4': 'b'}
        ]
        y = [0.5, 0.6]
        c1 = HyperParameter(ParamTypes.INT, [1, 5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001, 0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2, 8])
        c4 = HyperParameter(ParamTypes.STRING, ['a', 'b', 'c'])
        tunables = [('t1', c1), ('t2', c2), ('t3', c3), ('t4', c4)]
        u = Uniform(tunables)
        u.add(X, y)
        u.add({'t1': 3.5, 't2': 0.1, 't3': 3.2, 't4': 'a'}, 0.8)
        for i in range(100):
            proposed = u.propose()
            self.assertTrue(proposed['t1'] >= 1 and proposed['t1'] <= 5)
            self.assertTrue(proposed['t2'] >= 0.0001 and proposed['t2'] <= 0.1)
            self.assertTrue(proposed['t3'] >= 2 and proposed['t3'] <= 8)
            self.assertTrue(proposed['t4'] in ['a', 'b', 'c'])
        multi_proposed = u.propose(10)
        for proposed in multi_proposed:
            self.assertTrue(proposed['t1'] >= 1 and proposed['t1'] <= 5)
            self.assertTrue(proposed['t2'] >= 0.0001 and proposed['t2'] <= 0.1)
            self.assertTrue(proposed['t3'] >= 2 and proposed['t3'] <= 8)
            self.assertTrue(proposed['t4'] in ['a', 'b', 'c'])

    def test_gp(self):
        X = [{'a': 1.1, 'b': 0.01, 'c': 3.5}, {'a': 4, 'b': 0.001, 'c': 6.2}]
        y = [0.5, 0.6]
        c1 = HyperParameter(ParamTypes.INT, [1, 5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001, 0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2, 8])
        tunables = [('a', c1), ('b', c2), ('c', c3)]
        u = GP(tunables)
        u.add(X, y)
        for i in range(100):
            proposed = u.propose()
            self.assertTrue(proposed['a'] >= 1 and proposed['a'] <= 5)
            self.assertTrue(proposed['b'] >= 0.0001 and proposed['b'] <= 0.1)
            self.assertTrue(proposed['c'] >= 2 and proposed['c'] <= 8)

    def test_gpei(self):
        X = [{'a': 1.1, 'b': 0.01, 'c': 3.5}, {'a': 4, 'b': 0.001, 'c': 6.2}]
        y = [0.5, 0.6]
        c1 = HyperParameter(ParamTypes.INT, [1, 5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001, 0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2, 8])
        tunables = [('a', c1), ('b', c2), ('c', c3)]
        u = GPEi(tunables)
        u.add(X, y)
        for i in range(100):
            proposed = u.propose()
            self.assertTrue(proposed['a'] >= 1 and proposed['a'] <= 5)
            self.assertTrue(proposed['b'] >= 0.0001 and proposed['b'] <= 0.1)
            self.assertTrue(proposed['c'] >= 2 and proposed['c'] <= 8)

    def test_gpeivelocity(self):
        X = [{'a': 1.1, 'b': 0.01, 'c': 3.5}, {'a': 4, 'b': 0.001, 'c': 6.2}]
        y = [0.5, 0.6]
        c1 = HyperParameter(ParamTypes.INT, [1, 5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001, 0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2, 8])
        tunables = [('a', c1), ('b', c2), ('c', c3)]
        u = GPEiVelocity(tunables)
        u.add(X, y)
        for i in range(100):
            proposed = u.propose()
            self.assertTrue(proposed['a'] >= 1 and proposed['a'] <= 5)
            self.assertTrue(proposed['b'] >= 0.0001 and proposed['b'] <= 0.1)
            self.assertTrue(proposed['c'] >= 2 and proposed['c'] <= 8)

    def test_gcp(self):
        X = [{'a': 1.1, 'b': 0.01, 'c': 3.5}, {'a': 4, 'b': 0.001, 'c': 6.2}]
        y = [0.5, 0.6]
        c1 = HyperParameter(ParamTypes.INT, [1, 5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001, 0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2, 8])
        tunables = [('a', c1), ('b', c2), ('c', c3)]
        u = GCP(tunables)
        u.add(X, y)
        proposed = u.propose()
        self.assertTrue(proposed['a'] >= 1 and proposed['a'] <= 5)
        self.assertTrue(proposed['b'] >= 0.0001 and proposed['b'] <= 0.1)
        self.assertTrue(proposed['c'] >= 2 and proposed['c'] <= 8)

    def test_gcpei(self):
        X = [{'a': 1.1, 'b': 0.01, 'c': 3.5}, {'a': 4, 'b': 0.001, 'c': 6.2}]
        y = [0.5, 0.6]
        c1 = HyperParameter(ParamTypes.INT, [1, 5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001, 0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2, 8])
        tunables = [('a', c1), ('b', c2), ('c', c3)]
        u = GCPEi(tunables)
        u.add(X, y)
        proposed = u.propose()
        self.assertTrue(proposed['a'] >= 1 and proposed['a'] <= 5)
        self.assertTrue(proposed['b'] >= 0.0001 and proposed['b'] <= 0.1)
        self.assertTrue(proposed['c'] >= 2 and proposed['c'] <= 8)

    def test_gcpeivelocity(self):
        X = [{'a': 1.1, 'b': 0.01, 'c': 3.5}, {'a': 4, 'b': 0.001, 'c': 6.2}]
        y = [0.5, 0.6]
        c1 = HyperParameter(ParamTypes.INT, [1, 5])
        c2 = HyperParameter(ParamTypes.FLOAT_EXP, [0.0001, 0.1])
        c3 = HyperParameter(ParamTypes.FLOAT, [2, 8])
        tunables = [('a', c1), ('b', c2), ('c', c3)]
        u = GCPEiVelocity(tunables)
        u.add(X, y)
        proposed = u.propose()
        self.assertTrue(proposed['a'] >= 1 and proposed['a'] <= 5)
        self.assertTrue(proposed['b'] >= 0.0001 and proposed['b'] <= 0.1)
        self.assertTrue(proposed['c'] >= 2 and proposed['c'] <= 8)


if __name__ == '__main__':
    unittest.main()
