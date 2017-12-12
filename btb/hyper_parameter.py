from collections import namedtuple

class ParamTypes:
	INT = "int"
	INT_EXP = "int_exp"
	INT_CAT = "int_cat"
	FLOAT = "float"
	FLOAT_EXP = "float_exp"
	STRING = "string"
	BOOL = "bool"

# List of exponential hyperparameter types
EXP_TYPES = [ParamTypes.INT_EXP, ParamTypes.FLOAT_EXP]

# List of categorical hyperparameter types
CAT_TYPES = [ParamTypes.INT_CAT, ParamTypes.STRING, ParamTypes.BOOL]


# our HyperParameter object
class HyperParameter(object):
    def __init__(self, typ, rang):
        for i in range(len(rang)):
            if typ in [ParamTypes.INT, ParamTypes.INT_EXP,
                       ParamTypes.INT_CAT]:
                rang[i] = int(rang[i])
            elif typ in [ParamTypes.FLOAT, ParamTypes.FLOAT_EXP]:
                rang[i] = float(rang[i])
            elif typ == ParamTypes.STRING:
                rang[i] = str(rang[i])
            elif typ == ParamTypes.BOOL:
                rang[i] = bool(rang[i])
        self.type = typ
        self.range = rang

    @property
    def is_exponential(self):
        return self.type in EXP_TYPES

    @property
    def is_categorical(self):
        return self.type in CAT_TYPES
