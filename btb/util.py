import random


def shuffle(iterable):
    iterable = list(iterable)
    inds = list(range(len(iterable)))
    random.shuffle(inds)
    for i in inds:
        yield iterable[i]
