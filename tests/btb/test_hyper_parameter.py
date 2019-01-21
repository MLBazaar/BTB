import copy
import pickle
import random
import unittest

import numpy as np
from mock import patch

from btb.hyper_parameter import HyperParameter, ParamTypes


class TestHyperparameter(unittest.TestCase):

    def setUp(self):
        self.parameter_constructions = [
            (ParamTypes.INT, [1, 3]),
            (ParamTypes.INT_EXP, [10, 10000]),
            (ParamTypes.INT_CAT, [None, 10, 10000]),
            (ParamTypes.FLOAT, [1.5, 3.2]),
            (ParamTypes.FLOAT_EXP, [0.001, 100]),
            (ParamTypes.FLOAT_CAT, [None, 0.1, 0.6, 0.5]),
            (ParamTypes.BOOL, [None, True, False]),
            (ParamTypes.STRING, [None, 'a', 'b', 'c']),
        ]

    @patch('btb.hyper_parameter.HyperParameter.subclasses')
    def test_cast_not_implemented(self, subclasses_mock):

        # create a fake HyperParameter subclass
        class FakeHyperParameter(HyperParameter):
            param_type = ParamTypes.INT

        # Make FakeHyperParameter the only subclass for this test
        subclasses_mock.return_value = [FakeHyperParameter]

        with self.assertRaises(NotImplementedError):
            HyperParameter(ParamTypes.INT, [1])

    def test___init___with_string_param_type_valid(self):
        r = random.Random()
        r.seed(1)

        def random_case(s):
            return ''.join(
                r.choice([str.upper, str.lower])(c)
                for c in s
            )

        # allowed string param types
        for type_, range_ in self.parameter_constructions:
            for recase in [str.upper, str.lower, random_case]:
                str_type = recase(type_.name)
                self.assertEqual(
                    HyperParameter(str_type, range_),
                    HyperParameter(type_, range_)
                )

    def test___init___with_unicode_param_type(self):
        param_type_str = 'int'
        param_type_unicode = u'int'
        param_range = [0, 10]

        self.assertEqual(
            HyperParameter(param_type_unicode, param_range),
            HyperParameter(param_type_str, param_range))

    def test___init___with_string_param_type_invalid(self):
        # invalid string param types
        invalid_param_types = ['a', 0, object(), 'integer', 'foo']
        for invalid_param_type in invalid_param_types:
            with self.assertRaises(ValueError):
                HyperParameter(invalid_param_type, [None])

    def test___copy__(self):
        for typ, rang in self.parameter_constructions:
            hyp = HyperParameter(typ, rang)
            hyp_copy = copy.copy(hyp)
            self.assertIsNot(hyp, hyp_copy)
            self.assertIs(type(hyp), type(hyp_copy))
            self.assertEqual(hyp.range, hyp_copy.range)

            # shallow copy should just have copied references
            self.assertIs(hyp.range, hyp_copy.range)

    def test___deepcopy__(self):
        for typ, rang in self.parameter_constructions:
            hyp = HyperParameter(typ, rang)
            hyp_copy = copy.deepcopy(hyp)
            self.assertIsNot(hyp, hyp_copy)
            self.assertIs(type(hyp), type(hyp_copy))
            self.assertEqual(hyp.range, hyp_copy.range)

            # deep copy should have new attributes
            self.assertIsNot(hyp.range, hyp_copy.range)

    def test___eq__not_implemented(self):
        an_hyperparam = HyperParameter(ParamTypes.INT, [1, 5])
        not_an_hyperparam = 3

        self.assertEqual(an_hyperparam.__eq__(not_an_hyperparam), NotImplemented)
        self.assertNotEqual(an_hyperparam, not_an_hyperparam)

    def test_can_pickle(self):
        for protocol in range(0, pickle.HIGHEST_PROTOCOL + 1):
            for param_type, param_range in self.parameter_constructions:
                param = HyperParameter(param_type, param_range)
                pickled = pickle.dumps(param, protocol)
                unpickled = pickle.loads(pickled)
                self.assertEqual(param, unpickled)

    # ############## #
    # Specific Types #
    # ############## #

    def test_int(self):
        hyp = HyperParameter(ParamTypes.INT, [None, 1, 3])
        self.assertEqual(hyp.range, [None, 1, 3])
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
        hyp = HyperParameter(ParamTypes.FLOAT, [None, 1.5, 3.2])
        self.assertEqual(hyp.range, [None, 1.5, 3.2])
        transformed = hyp.fit_transform(
            np.array([2, 2.4, 3.1]),
            np.array([0.5, 0.6, 0.1]),
        )
        np.testing.assert_array_equal(transformed, np.array([2, 2.4, 3.1]))
        inverse_transform = hyp.inverse_transform([1.7])
        np.testing.assert_array_equal(inverse_transform, np.array([1.7]))

    def test_float_exp(self):
        hyp = HyperParameter(ParamTypes.FLOAT_EXP, [None, 0.001, 100])
        self.assertEqual(hyp.range, [None, -3.0, 2.0])
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
        hyp = HyperParameter(ParamTypes.INT_EXP, [None, 10, 10000])
        self.assertEqual(hyp.range, [None, 1, 4])
        transformed = hyp.fit_transform(np.array([100]), np.array([0.5]))
        np.testing.assert_array_equal(transformed, np.array([2]))
        inverse_transform = hyp.inverse_transform([3])
        np.testing.assert_array_equal(inverse_transform, np.array([1000]))

    def test_int_cat(self):
        hyp = HyperParameter(ParamTypes.INT_CAT, [None, 10, 10000])
        transformed = hyp.fit_transform(
            np.array([None, 10, 10000]),
            np.array([0.1, 0.5, 0.9])
        )
        np.testing.assert_array_equal(transformed, np.array([0.1, 0.5, 0.9]))
        inverse_transform = hyp.inverse_transform([1.0, 0.0, 0.4, 0.8])
        np.testing.assert_array_equal(
            inverse_transform,
            np.array([10000, None, 10, 10000])
        )

    def test_float_cat(self):
        hyp = HyperParameter(ParamTypes.FLOAT_CAT, [None, 0.1, 0.6, 0.5])
        transformed = hyp.fit_transform(
            np.array([None, 0.1, 0.6, 0.1, 0.6]),
            np.array([1.0, 0.1, 0.2, 0.3, 0.4])
        )
        np.testing.assert_allclose(
            transformed,
            np.array([1.0, 0.2, 0.3, 0.2, 0.3])
        )
        self.assertEqual(hyp.range, [0.0, 1.0])
        inverse_transform = hyp.inverse_transform([0.3, 0.0, 0.1, 0.5, 0.27, 0.9])
        np.testing.assert_array_equal(
            inverse_transform,
            np.array([0.6, 0.5, 0.1, 0.6, 0.6, None])
        )

    def test_bool(self):
        hyp = HyperParameter(ParamTypes.BOOL, [None, True, False])
        transformed = hyp.fit_transform(
            np.array([None, True, False]),
            np.array([0.1, 0.5, 0.7])
        )
        np.testing.assert_array_equal(transformed, np.array([0.1, 0.5, 0.7]))
        self.assertEqual(hyp.range, [0.1, 0.7])
        inverse_transform = hyp.inverse_transform([0.2, 0.7, 0.6, 0.5])
        np.testing.assert_array_equal(
            inverse_transform,
            np.array([None, False, False, True])
        )

    def test_string(self):
        hyp = HyperParameter(ParamTypes.STRING, [None, 'a', 'b', 'c'])
        transformed = hyp.fit_transform(
            np.array([None, 'a', 'b', 'c']),
            np.array([0.1, 0.3, 0.6, 0.9])
        )
        self.assertEqual(hyp.range, [0.1, 0.9])
        np.testing.assert_array_equal(
            transformed,
            np.array([0.1, 0.3, 0.6, 0.9])
        )
        inverse_transform = hyp.inverse_transform(np.array([0.0, 0.4, 0.5, 0.8]))
        np.testing.assert_array_equal(
            inverse_transform,
            np.array([None, 'a', 'b', 'c'])
        )

        inverse_transform = hyp.inverse_transform([3])
        np.testing.assert_array_equal(inverse_transform, np.array(['c']))
