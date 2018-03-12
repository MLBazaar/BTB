from builtins import object, str as newstr
from collections import namedtuple
import numpy as np
import math

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
		sub_classes = HyperParameter.__subclasses__()[:]
		if cls is HyperParameter:
			i = 0
			while i < len(sub_classes):
				sub_cls = sub_classes[i]
				if sub_cls.is_type_for(typ):
					return super(HyperParameter, cls).__new__(sub_cls)
				sub_classes.extend(sub_cls.__subclasses__())
				i += 1
		else:
			return super(HyperParameter, cls).__new__(cls)
	def __init__(self, rang, cast):
		print("called")
		for i, val in enumerate(rang):
			if val is None:
				# the value None is allowed for every parameter type
				continue
			rang[i] = cast(val)
		self.rang = rang
	#Open Q: is this necessary (ie is mapping to linear space with knowledge)
	#of it being an integer necessary, or should we just round after?
	@property
	def is_integer(self):
		return False

	@classmethod
	def is_type_for(cls, typ):
		return False

	def fit_transform(self, x, y):
		return x

	#TODO: is x one value or np array? Seems easy to assume one value
	def inverse_transform(self, x):
		return x

class IntHyperParameter(HyperParameter):
	def __init__(self, typ, rang):
		print("in int")
		HyperParameter.__init__(self, rang, int)

	@classmethod
	def is_type_for(cls, typ):
		return typ == ParamTypes.INT

	@property
	def is_integer(self):
		return True

	def inverse_transform(self, x):
		return int(x)

class FloatHyperParameter(HyperParameter):
	def __init__(self, typ, rang):
		HyperParameter.__init__(self, rang, float)

	@classmethod
	def is_type_for(cls, typ):
		return typ == ParamTypes.FLOAT

class FloatExpHyperParameter(HyperParameter):
	def __init__(self, typ, rang):
		HyperParameter.__init__(self, rang, lambda x: math.log10(float(x)))

	@classmethod
	def is_type_for(cls, typ):
		return typ == ParamTypes.FLOAT_EXP

	#transfrom to log base 10(x)
	def fit_transform(self, x,y):
		return np.log10(x)

	def inverse_transform(self, x):
		return 10.0**x

class IntExpHyperParameter(FloatExpHyperParameter):
	def __init__(self, typ, rang):
		HyperParameter.__init__(self, rang, lambda x: math.log10(int(x)))

	@classmethod
	def is_type_for(cls, typ):
		return typ == ParamTypes.INT_EXP

	@property
	def is_integer(self):
		return True

	def inverse_transform(self, x):
		return int(FloatExpHyperParameter.inverse_transform(self, x))

class CatHyperParameter(HyperParameter):
#Open Q: shoudl the search space always be 0-1? Or should it be
#min, max of values in cat_transform after fit transform?
	def __init__(self, rang, cast):
		self.cat_transform = {cast(each): 0 for each in rang}
		HyperParameter.__init__(self, [0.0, 1.0], float)

	def fit_transform(self, x, y):
		self.cat_transform = {each: (0,0) for each in self.cat_transform}
		for i in range(len(x)):
			self.cat_transform[x[i]] = (
				self.cat_transform[x[i]][0] + y[i],
				self.cat_transform[x[i]][1]+1
			)
		for key, value in self.cat_transform:
			self.cat_transform[key] = value[0]/float(value[1])
		return np.vectorize(self.cat_transform.get)(x)

	def inverse_transform(self, x):
		#TODO deal with repeated values
		inv_map = {v: k for k, v in self.cat_transform.items()}
		def invert(inv_map, x):
			nearest = np.find_nearest(np.array(inv_map.keys()), x)
			return np.vectorize(inv_map.get)(nearest)
		return np.vectorize(invert)(inv_map, x)

class IntCatHyperParameter(CatHyperParameter):
	def __init__(self, typ, rang):
		CatHyperParameter.__init__(self, rang, int)

	@classmethod
	def is_type_for(cls, typ):
		return typ == ParamTypes.INT_CAT

class FloatCatHyperParameter(CatHyperParameter):
	def __init__(self, typ, rang):
		CatHyperParameter.__init__(self, rang, float)

	@classmethod
	def is_type_for(cls, typ):
		return typ == ParamTypes.FLOAT_CAT

class StringCatHyperParameter(CatHyperParameter):
	def __init__(self, typ, rang):
		CatHyperParameter.__init__(self, rang, lambda x: str(newstr(x)))

	@classmethod
	def is_type_for(cls, typ):
		return typ == ParamTypes.STRING

class BoolCatHyperParameter(CatHyperParameter):
	def __init__(self, typ, rang):
		CatHyperParameter.__init__(self, rang, bool)

	@classmethod
	def is_type_for(cls, typ):
		return typ == ParamTypes.BOOL
