from builtins import object, str as newstr
from collections import namedtuple

class ParamTypes(object):
	INT = "int"
	INT_EXP = "int_exp"
	FLOAT = "float"
	FLOAT_EXP = "float_exp"

# List of exponential hyperparameter types
EXP_TYPES = [ParamTypes.INT_EXP, ParamTypes.FLOAT_EXP]


# our HyperParameter object
class HyperParameter(object):
    def __init__(self, typ, rang):
        for i, val in enumerate(rang):
            if val is None:
                # the value None is allowed for every parameter type
                continue
            if typ in [ParamTypes.INT, ParamTypes.INT_EXP]:
                rang[i] = int(val)
            elif typ in [ParamTypes.FLOAT, ParamTypes.FLOAT_EXP]:
                rang[i] = float(val)
        self.type = typ
        self.range = rang

    @property
    def is_exponential(self):
        return self.type in EXP_TYPES
