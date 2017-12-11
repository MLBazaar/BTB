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
    def __init__(self, type, range):
        self.type = type
        self.range = range

    @property
    def is_exponential(self):
        return self.type in EXP_TYPES

    @property
    def is_categorical(self):
        return self.type in CAT_TYPES
