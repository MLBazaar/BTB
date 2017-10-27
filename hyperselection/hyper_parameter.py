from collections import namedtuple

# our HyperParameter named tuple
HyperParameter = namedtuple('HyperParameter', 'range type is_categorical')

class ParamTypes:
	INT = "INT"
	INT_EXP = "INT_EXP"
	FLOAT = "FLOAT"
	FLOAT_EXP = "FLOAT_EXP"
	STRING = "STRING"
	BOOL = "BOOL"

