import math
import random
from collections import Iterable, defaultdict
from enum import Enum

import numpy as np
import six


class ParamTypes(Enum):
    INT = 1
    INT_EXP = 2
    INT_CAT = 3
    FLOAT = 4
    FLOAT_EXP = 5
    FLOAT_CAT = 6
    STRING = 7
    BOOL = 8


class HyperParameter(object):

    param_type = None
    is_integer = False

    _subclasses = []

    @classmethod
    def _get_subclasses(cls):
        subclasses = []
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass._get_subclasses())

        return subclasses

    @classmethod
    def subclasses(cls):
        if not cls._subclasses:
            cls._subclasses = cls._get_subclasses()

        return cls._subclasses

    def __new__(cls, param_type=None, param_range=None):
        if not isinstance(param_type, ParamTypes):
            if (isinstance(param_type, six.string_types) and
                    param_type.upper() in ParamTypes.__members__):
                param_type = ParamTypes[param_type.upper()]
            else:
                raise ValueError('Invalid param type {}'.format(param_type))

        for subclass in cls.subclasses():
            if subclass.param_type is param_type:
                return super(HyperParameter, cls).__new__(subclass)

    def cast(self, value):
        raise NotImplementedError()

    def __init__(self, param_type=None, param_range=None):

        # maintain original param_range
        self._param_range = param_range

        self.range = [self.cast(value) for value in param_range]

    def fit_transform(self, x, y):
        return x

    def inverse_transform(self, x):
        return x

    def get_grid_axis(self, grid_size):
        if self.is_integer:
            return np.round(
                np.linspace(self.range[0], self.range[1], grid_size)
            )

        return np.round(
            # NOTE: flake8 reported "undefined name 'param'"
            # so "param" was repaced by "self".
            # Remove this after review.
            np.linspace(self.range[0], self.range[1], grid_size),
            decimals=5,
        )

    def __eq__(self, other):
        # See https://stackoverflow.com/a/25176504 for details
        if isinstance(self, other.__class__):
            return (self.param_type is other.param_type and
                    self.is_integer == other.is_integer and
                    self.range == other.range)

        return NotImplemented

    def __ne__(self, other):   # pragma: no cover
        # Not needed in Python 3
        # See https://stackoverflow.com/a/25176504 for details
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x

        return NotImplemented

    def __getnewargs__(self):
        return (self.param_type, self._param_range)


class IntHyperParameter(HyperParameter):
    param_type = ParamTypes.INT
    is_integer = True

    def cast(self, value):
        if value is not None:
            return int(value)
        else:
            return None

    def inverse_transform(self, x):
        return x.astype(int)


class FloatHyperParameter(HyperParameter):
    param_type = ParamTypes.FLOAT

    def cast(self, value):
        if value is not None:
            return float(value)
        else:
            return None


class FloatExpHyperParameter(HyperParameter):
    param_type = ParamTypes.FLOAT_EXP

    def cast(self, value):
        if value is not None:
            return math.log10(float(value))
        else:
            return None

    def fit_transform(self, x, y):
        x = x.astype(float)
        return np.log10(x)

    def inverse_transform(self, x):
        return np.power(10.0, x)


class IntExpHyperParameter(FloatExpHyperParameter):
    param_type = ParamTypes.INT_EXP
    is_integer = True

    def inverse_transform(self, x):
        return super(IntExpHyperParameter, self).inverse_transform(x).astype(int)


class CatHyperParameter(HyperParameter):

    def __init__(self, param_type=None, param_range=None):
        super(CatHyperParameter, self).__init__(param_type, param_range)
        self.cat_transform = {self.cast(each): 0 for each in param_range}
        self.range = [0.0, 1.0]

    def fit_transform(self, x, y):
        # Accumulate the scores of each category
        # and the number of times that we have used it
        tmp_cat_transform = {each: (0, 0) for each in self.cat_transform.keys()}
        for i in range(len(x)):
            tmp_cat_transform[x[i]] = (
                tmp_cat_transform[x[i]][0] + y[i],    # sum score
                tmp_cat_transform[x[i]][1] + 1        # count occurrences
            )

        # If we have at least one score, compute the average
        for key, value in tmp_cat_transform.items():
            if value[1] != 0:
                self.cat_transform[key] = value[0] / float(value[1])
            else:
                self.cat_transform[key] = 0

        # Compute the range using the min and max scores
        range_max = max(
            self.cat_transform.keys(),
            key=(lambda k: self.cat_transform[k])
        )

        range_min = min(
            self.cat_transform.keys(),
            key=(lambda k: self.cat_transform[k])
        )

        self.range = [
            self.cat_transform[range_min],
            self.cat_transform[range_max]
        ]

        return np.vectorize(self.cat_transform.get)(x)

    def inverse_transform(self, x):
        # Compute the inverse dictionary
        inv_map = defaultdict(list)
        for key, value in self.cat_transform.items():
            inv_map[value].append(key)

        keys = np.fromiter(inv_map.keys(), dtype=float)

        def invert(x):
            diff = (np.abs(keys - x))
            min_diff = diff[0]
            max_key = keys[0]

            # Find the score which is closer to the given value
            for i in range(len(diff)):
                if diff[i] < min_diff:
                    min_diff = diff[i]
                    max_key = keys[i]
                elif diff[i] == min_diff and keys[i] > max_key:
                    min_diff = diff[i]
                    max_key = keys[i]

            # Get a random category from the ones that had the given score
            return random.choice(np.vectorize(inv_map.get)(max_key))

        if isinstance(x, Iterable):
            transformed = list(map(invert, x))
            if isinstance(x, np.ndarray):
                transformed = np.array(transformed)

        else:
            transformed = invert(x)

        return transformed


class IntCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.INT_CAT

    def cast(self, value):
        if value is not None:
            return int(value)
        else:
            return None


class FloatCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.FLOAT_CAT

    def cast(self, value):
        if value is not None:
            return float(value)
        else:
            return None


class StringCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.STRING

    def cast(self, value):
        if value is not None:
            return str(value)
        else:
            return None


class BoolCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.BOOL

    def cast(self, value):
        if value is not None:
            return bool(value)
        else:
            return None
