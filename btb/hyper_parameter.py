from builtins import object, str as newstr
from collections import namedtuple, defaultdict
import random
import numpy as np
import math
import operator

class ParamTypes(object):
	INT = "int"
	INT_EXP = "int_exp"
	INT_CAT = "int_cat"
	FLOAT = "float"
	FLOAT_EXP = "float_exp"
	FLOAT_CAT = "float_cat"
	STRING = "string"
	BOOL = "bool"

#HyperParameter object
class HyperParameter(object):
	def __new__(cls, typ, rang):
		class_generator = {
			ParamTypes.INT: IntHyperParameter,
			ParamTypes.INT_EXP: IntExpHyperParameter,
			ParamTypes.INT_CAT: IntCatHyperParameter,
			ParamTypes.FLOAT: FloatHyperParameter,
			ParamTypes.FLOAT_EXP: FloatExpHyperParameter,
			ParamTypes.FLOAT_CAT: FloatCatHyperParameter,
			ParamTypes.STRING: StringCatHyperParameter,
			ParamTypes.BOOL: BoolCatHyperParameter,
		}
		if cls is HyperParameter:
			return super(HyperParameter, cls).__new__(
				class_generator[typ]
			)
		else:
			return super(HyperParameter, cls).__new__(cls)

	def __init__(self, rang, cast):
		for i, val in enumerate(rang):
			if val is None:
				# the value None is allowed for every parameter type
				continue
			rang[i] = cast(val)
		self.range = rang

	@property
	def is_integer(self):
		return False

	def fit_transform(self, x, y):
		return x

	def inverse_transform(self, x):
		return x

class IntHyperParameter(HyperParameter):
	def __init__(self, typ, rang):
		HyperParameter.__init__(self, rang, int)

	@property
	def is_integer(self):
		return True

	def inverse_transform(self, x):
		return int(x)

class FloatHyperParameter(HyperParameter):
	def __init__(self, typ, rang):
		HyperParameter.__init__(self, rang, float)

class FloatExpHyperParameter(HyperParameter):
	def __init__(self, typ, rang):
		HyperParameter.__init__(self, rang, lambda x: math.log10(float(x)))

	def fit_transform(self, x,y):
		x = x.astype(float)
		return np.log10(x)

	def inverse_transform(self, x):
		return np.power(10.0, x)

class IntExpHyperParameter(FloatExpHyperParameter):
	def __init__(self, typ, rang):
		HyperParameter.__init__(self, rang, lambda x: math.log10(int(x)))

	@property
	def is_integer(self):
		return True

	def inverse_transform(self, x):
		return np.astype(FloatExpHyperParameter.inverse_transform(self, x), int)

class CatHyperParameter(HyperParameter):
	def __init__(self, rang, cast):
		self.cat_transform = {cast(each): 0 for each in rang}
		#this is a dummy range until the transformer is fit
		HyperParameter.__init__(self, [0.0, 1.0], float)

	def fit_transform(self, x, y):
		self.cat_transform = {each: (0,0) for each in self.cat_transform}
		for i in range(len(x)):
			self.cat_transform[x[i]] = (
				self.cat_transform[x[i]][0] + y[i],
				self.cat_transform[x[i]][1]+1
			)
		for key, value in self.cat_transform.items():
			if value[1] != 0:
				self.cat_transform[key] = value[0]/float(value[1])
			else:
				self.cat_transform[key] = 0
		rang_max = max(self.cat_transform.keys(), key=(lambda k: self.cat_transform[k]))
		rang_min = min(self.cat_transform.keys(), key=(lambda k: self.cat_transform[k]))
		self.range = [self.cat_transform[rang_min], self.cat_transform[rang_max]]
		return np.vectorize(self.cat_transform.get)(x)

	def inverse_transform(self, x):
		inv_map = defaultdict(list)
		for key, value in self.cat_transform.items():
			inv_map[value].append(key)

		def invert(inv_map, x):
			keys = np.fromiter(inv_map.keys(), dtype=float)
			diff = (np.abs(keys-x))
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
		return np.vectorize(invert)(inv_map, x)

class IntCatHyperParameter(CatHyperParameter):
	def __init__(self, typ, rang):
		CatHyperParameter.__init__(self, rang, int)

class FloatCatHyperParameter(CatHyperParameter):
	def __init__(self, typ, rang):
		CatHyperParameter.__init__(self, rang, float)

class StringCatHyperParameter(CatHyperParameter):
	def __init__(self, typ, rang):
		CatHyperParameter.__init__(self, rang, lambda x: str(newstr(x)))

class BoolCatHyperParameter(CatHyperParameter):
	def __init__(self, typ, rang):
		CatHyperParameter.__init__(self, rang, bool)
