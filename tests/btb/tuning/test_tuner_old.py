import unittest

import numpy as np

from btb import HyperParameter, ParamTypes
from btb.tuning.gcp import GCP, GCPEi, GCPEiVelocity
from btb.tuning.gp import GP, GPEi, GPEiVelocity
from btb.tuning.tuner import BaseTuner
from btb.tuning.uniform import Uniform


class TestTuner(unittest.TestCase):
    def setUp(self):
        self.tunables = [
            ('t1', HyperParameter(ParamTypes.INT, [1, 3])),
            ('t2', HyperParameter(ParamTypes.INT_EXP, [10, 10000])),
            ('t3', HyperParameter(ParamTypes.FLOAT, [1.5, 3.2])),
            ('t4', HyperParameter(ParamTypes.FLOAT_EXP, [0.001, 100])),
            ('t5', HyperParameter(ParamTypes.FLOAT_CAT, [0.1, 0.6, 0.5])),
            ('t6', HyperParameter(ParamTypes.BOOL, [True, False])),
            ('t7', HyperParameter(ParamTypes.STRING, ['a', 'b', 'c'])),
        ]

        self.X = [
            {'t1': 2, 't2': 1000, 't3': 3.0, 't4': 0.1, 't5': 0.5,
                't6': True, 't7': 'a'},
            {'t1': 1, 't2': 100, 't3': 1.9, 't4': 0.1, 't5': 0.6,
                't6': True, 't7': 'b'},
            {'t1': 3, 't2': 10, 't3': 2.6, 't4': 0.01, 't5': 0.1,
                't6': False, 't7': 'c'},
        ]

        self.Y = [0.5, 0.6, 0.1]

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

    def test_add_multitype(self):
        t = BaseTuner(self.tunables)
        t.add(self.X, self.Y)
        X = np.array([
            [2, 3.0, 3.0, -1.0, 0.5, 0.55, 0.5],
            [1, 2.0, 1.9, -1.0, 0.6, 0.55, 0.6],
            [3, 1.0, 2.6, -2.0, 0.1, 0.1, 0.1],
        ], dtype=object)
        np.testing.assert_array_equal(t.X, X)
        self.assertEqual(t.X.dtype, np.float64)
