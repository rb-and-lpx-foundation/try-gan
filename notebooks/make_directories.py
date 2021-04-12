import os


def make_directory(d, sub=None):
    if sub is not None:
        d = os.path.join(d, sub)
    if not os.path.exists(d):
        os.mkdir(d)
    return os.path.abspath(d)


def make_directories(base, sub=None):
    base = make_directory(base, sub)
    train = make_directory(base, "train")
    test = make_directory(base, "test")
    return base, test, train


def get_regime(base):
    base = make_directory(base)
    data = make_directories(base, "data")
    cache = make_directories(base, "cache")
    log = make_directory(base, "log")
    check = make_directory(base, "checkpoint")
    return base, data, cache, log, check
