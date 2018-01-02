from builtins import object, str as newstr
from collections import namedtuple

class ParamTypes(object):
	INT = "int"
	INT_EXP = "int_exp"
	INT_CAT = "int_cat"
	FLOAT = "float"
	FLOAT_EXP = "float_exp"
	FLOAT_CAT = "float_cat"
	STRING = "string"
	BOOL = "bool"

# List of exponential hyperparameter types
EXP_TYPES = [ParamTypes.INT_EXP, ParamTypes.FLOAT_EXP]

# List of categorical hyperparameter types
CAT_TYPES = [ParamTypes.INT_CAT, ParamTypes.FLOAT_CAT, ParamTypes.STRING,
             ParamTypes.BOOL]


# our HyperParameter object
class HyperParameter(object):
    def __init__(self, typ, rang):
        for i, val in enumerate(rang):
            if val is None:
                # the value None is allowed for every parameter type
                continue
            if typ in [ParamTypes.INT, ParamTypes.INT_EXP,
                       ParamTypes.INT_CAT]:
                rang[i] = int(val)
            elif typ in [ParamTypes.FLOAT, ParamTypes.FLOAT_EXP,
                         ParamTypes.FLOAT_CAT]:
                rang[i] = float(val)
            elif typ == ParamTypes.STRING:
                rang[i] = str(newstr(val))
            elif typ == ParamTypes.BOOL:
                rang[i] = bool(val)
        self.type = typ
        self.range = rang

    @property
    def is_exponential(self):
        return self.type in EXP_TYPES

    @property
    def is_categorical(self):
        return self.type in CAT_TYPES
