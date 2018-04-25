import copy
import unittest

import numpy as np

from btb.hyper_parameter import HyperParameter, ParamTypes


class TestHyperparameter(unittest.TestCase):
    def setUp(self):
        self.parameter_constructions = [
            (ParamTypes.INT, [1, 3]),
            (ParamTypes.INT_EXP, [10, 10000]),
            (ParamTypes.FLOAT, [1.5, 3.2]),
            (ParamTypes.FLOAT_EXP, [0.001, 100]),
            (ParamTypes.FLOAT_CAT, [0.1, 0.6, 0.5]),
            (ParamTypes.BOOL, [True, False]),
            (ParamTypes.STRING, ['a', 'b', 'c']),
        ]

    def test_int(self):
        hyp = HyperParameter(ParamTypes.INT, [1, 3])
        self.assertEqual(hyp.range, [1, 3])
        transformed = hyp.fit_transform(
            np.array([1, 2, 3]),
            np.array([0.5, 0.6, 0.1])
        )
        np.testing.assert_array_equal(transformed, np.array([1, 2, 3]))
        inverse_transform = hyp.inverse_transform(np.array([0.5]))
        np.testing.assert_array_equal(inverse_transform, np.array([0]))
        hyp = HyperParameter(ParamTypes.INT, [1.0, 3.0])
        self.assertEqual(hyp.range, [1, 3])

    def test_float(self):
        hyp = HyperParameter(ParamTypes.FLOAT, [1.5, 3.2])
        self.assertEqual(hyp.range, [1.5, 3.2])
        transformed = hyp.fit_transform(
            np.array([2, 2.4, 3.1]),
            np.array([0.5, 0.6, 0.1]),
        )
        np.testing.assert_array_equal(transformed, np.array([2, 2.4, 3.1]))
        inverse_transform = hyp.inverse_transform([1.7])
        np.testing.assert_array_equal(inverse_transform, np.array([1.7]))

    def test_float_exp(self):
        hyp = HyperParameter(ParamTypes.FLOAT_EXP, [0.001, 100])
        self.assertEqual(hyp.range, [-3.0, 2.0])
        transformed = hyp.fit_transform(
            np.array([0.01, 1, 10]),
            np.array([-2.0, 0.0, 1.0])
        )
        np.testing.assert_array_equal(
            transformed,
            np.array([-2.0, 0.0, 1.0])
        )
        inverse_transform = hyp.inverse_transform([-1.0])
        np.testing.assert_array_equal(inverse_transform, np.array([0.1]))
        inverse_transform = hyp.inverse_transform([1.0])
        np.testing.assert_array_equal(inverse_transform, np.array([10.0]))

    def test_int_exp(self):
        hyp = HyperParameter(ParamTypes.INT_EXP, [10, 10000])
        self.assertEqual(hyp.range, [1, 4])
        transformed = hyp.fit_transform(np.array([100]), np.array([0.5]))
        np.testing.assert_array_equal(transformed, np.array([2]))
        inverse_transform = hyp.inverse_transform([3])
        np.testing.assert_array_equal(inverse_transform, np.array([1000]))

    def test_int_cat(self):
        hyp = HyperParameter(ParamTypes.INT_CAT, [10, 10000])
        transformed = hyp.fit_transform(
            np.array([10, 10000]),
            np.array([1, 2])
        )
        np.testing.assert_array_equal(transformed, np.array([1, 2]))
        inverse_transform = hyp.inverse_transform([3, 0, 1, 2])
        np.testing.assert_array_equal(
            inverse_transform,
            np.array([10000, 10, 10, 10000])
        )

    def test_float_cat(self):
        hyp = HyperParameter(ParamTypes.FLOAT_CAT, [0.1, 0.6, 0.5])
        transformed = hyp.fit_transform(
            np.array([0.1, 0.6, 0.1, 0.6]),
            np.array([1, 2, 3, 4])
        )
        np.testing.assert_array_equal(
            transformed,
            np.array([2.0, 3.0, 2.0, 3.0])
        )
        self.assertEqual(hyp.range, [0.0, 3.0])
        inverse_transform = hyp.inverse_transform([3, 0, 1, 5, 2.5])
        np.testing.assert_array_equal(
            inverse_transform,
            np.array([0.6, 0.5, 0.1, 0.6, 0.6])
        )

    def test_bool(self):
        hyp = HyperParameter(ParamTypes.BOOL, [True, False])
        transformed = hyp.fit_transform(
            np.array([True, False]),
            np.array([0.5, 0.7])
        )
        np.testing.assert_array_equal(transformed, np.array([0.5, 0.7]))
        self.assertEqual(hyp.range, [0.5, 0.7])
        inverse_transform = hyp.inverse_transform([0.7, 0.6, 0.5])
        np.testing.assert_array_equal(
            inverse_transform,
            np.array([False, False, True])
        )

    def test_string(self):
        hyp = HyperParameter(ParamTypes.STRING, ['a', 'b', 'c'])
        transformed = hyp.fit_transform(
            np.array(['a', 'b', 'c']),
            np.array([2, 1, 3])
        )
        self.assertEqual(hyp.range, [1.0, 3.0])
        np.testing.assert_array_equal(
            transformed,
            np.array([2.0, 1.0, 3.0])
        )
        inverse_transform = hyp.inverse_transform([3])
        np.testing.assert_array_equal(inverse_transform, np.array(['c']))

    def test_copy(self):
        for typ, rang in self.parameter_constructions:
            hyp = HyperParameter(typ, rang)
            hyp_copy = copy.copy(hyp)
            self.assertIsNot(hyp, hyp_copy)
            self.assertIs(type(hyp), type(hyp_copy))
            self.assertEqual(hyp.range, hyp_copy.range)

            # shallow copy should just have copied references
            self.assertIs(hyp.range, hyp_copy.range)

    def test_deepcopy(self):
        for typ, rang in self.parameter_constructions:
            hyp = HyperParameter(typ, rang)
            hyp_copy = copy.deepcopy(hyp)
            self.assertIsNot(hyp, hyp_copy)
            self.assertIs(type(hyp), type(hyp_copy))
            self.assertEqual(hyp.range, hyp_copy.range)

            # deep copy should have new attributes
            self.assertIsNot(hyp.range, hyp_copy.range)
