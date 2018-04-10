from btb import HyperParameter, ParamTypes
import numpy as np

hyp = HyperParameter(ParamTypes.INT, [1, 3])
print("hyp", hyp)
print("is int", hyp.is_integer)
