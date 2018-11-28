import random

import numpy as np


def shuffle(iterable):
    iterable = list(iterable)
    inds = list(range(len(iterable)))
    random.shuffle(inds)
    for i in inds:
        yield iterable[i]


def asarray2d(arr):
    """Cast to 2d array"""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr
