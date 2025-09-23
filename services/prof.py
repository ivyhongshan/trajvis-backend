# services/prof.py
import time
import logging
from functools import wraps

def timed(label):
    def deco(fn):
        @wraps(fn)
        def inner(*a, **kw):
            t0 = time.time()
            try:
                return fn(*a, **kw)
            finally:
                logging.info("STAGE %s took %.3fs", label, time.time() - t0)
        return inner
    return deco
