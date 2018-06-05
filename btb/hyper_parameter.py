import copy
import math
import random
from collections import OrderedDict
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
    is_discrete = False

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
            if (isinstance(param_type, str) and
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
        self.range = [
            self.cast(value)
            # "the value None is allowed for every parameter type"
            if value is not None else None
            for value in param_range
        ]

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
        if self.is_discrete:
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
        # See https://stackoverflow.com/a/25176504/2514228 for details
        if isinstance(self, other.__class__):
            return (self.param_type is other.param_type and
                    self.is_discrete == other.is_discrete and
                    self.range == other.range)
        return NotImplemented

    def __ne__(self, other):
        # Not needed in Python 3
        # See https://stackoverflow.com/a/25176504/2514228 for details
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented


class IntHyperParameter(HyperParameter):
    param_type = ParamTypes.INT
    is_discrete = True

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
    is_discrete = True

    def inverse_transform(self, x):
        return super(IntExpHyperParameter, self).inverse_transform(x).astype(int)


class CatHyperParameter(HyperParameter):
    is_discrete = True

    def __init__(self, param_type=None, param_range=None):
        self.categories = np.array(param_range)
        self.range = [0, len(param_range) - 1]

    def fit_transform(self, x, y):
        return np.vectorize(self.categories.tolist().index)(x)[()]

    def inverse_transform(self, x):
        x = x.astype(int)
        return np.vectorize(self.categories.item)(x)[()]


class IntCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.INT_CAT


class FloatCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.FLOAT_CAT


class StringCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.STRING


class BoolCatHyperParameter(CatHyperParameter):
    param_type = ParamTypes.BOOL
