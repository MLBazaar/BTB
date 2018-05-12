import copy
import math
import random
from collections import defaultdict
from enum import Enum

import numpy as np


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
            if (isinstance(param_type, str)
                    and param_type.upper() in ParamTypes.__members__):
                param_type = ParamTypes[param_type.upper()]
            else:
                raise ValueError('Invalid param type {}'.format(param_type))

        for subclass in cls.subclasses():
            if subclass.param_type is param_type:
                return super(HyperParameter, cls).__new__(subclass)

    def cast(self, value):
        raise NotImplementedError()

    def __init__(self, param_type=None, param_range=None):
        for i, value in enumerate(param_range):
            # the value None is allowed for every parameter type
            if value is not None:
                param_range[i] = self.cast(value)

        self.range = param_range

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls, self.param_type, self.range)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls, self.param_type, self.range)
        result.__dict__.update(self.__dict__)

        memo[id(self)] = result

        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

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


class IntHyperParameter(HyperParameter):
    param_type = ParamTypes.INT
    is_integer = True

    def cast(self, value):
        return int(value)

    def inverse_transform(self, x):
        return x.astype(int)


class FloatHyperParameter(HyperParameter):
    param_type = ParamTypes.FLOAT

    def cast(self, value):
        return float(value)


class FloatExpHyperParameter(HyperParameter):
    param_type = ParamTypes.FLOAT_EXP

    def cast(self, value):
        return math.log10(float(value))

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
        self.cat_transform = {self.cast(each): 0 for each in param_range}

        # this is a dummy range until the transformer is fit
        super(CatHyperParameter, self).__init__(param_type, [0.0, 1.0])

    def fit_transform(self, x, y):
        self.cat_transform = {each: (0, 0) for each in self.cat_transform}
        for i in range(len(x)):
            self.cat_transform[x[i]] = (
                self.cat_transform[x[i]][0] + y[i],
                self.cat_transform[x[i]][1] + 1
            )

        for key, value in self.cat_transform.items():
            if value[1] != 0:
                self.cat_transform[key] = value[0] / float(value[1])
            else:
                self.cat_transform[key] = 0

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
        inv_map = defaultdict(list)
        for key, value in self.cat_transform.items():
            inv_map[value].append(key)

        def invert(inv_map, x):
            keys = np.fromiter(inv_map.keys(), dtype=float)
            diff = (np.abs(keys - x))
            min_diff = diff[0]
            max_key = keys[0]
            for i in range(len(diff)):
                if diff[i] < min_diff:
                    min_diff = diff[i]
                    max_key = keys[i]
                elif diff[i] == min_diff and keys[i] > max_key:
                    min_diff = diff[i]
                    max_key = keys[i]

            return random.choice(np.vectorize(inv_map.get)(max_key))

        inv_trans = np.vectorize(invert)(inv_map, x)
        return inv_trans.item() if np.ndim(inv_trans) == 0 else inv_trans


class IntCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.INT_CAT

    def cast(self, value):
        return int(value)


class FloatCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.FLOAT_CAT

    def cast(self, value):
        return float(value)


class StringCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.STRING

    def cast(self, value):
        return str(value)


class BoolCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.BOOL

    def cast(self, value):
        return bool(value)
