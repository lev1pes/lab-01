"""Декоратор вимірювання часу зі збереженням метаданих функції."""

import logging
from functools import wraps
from time import perf_counter


def timed(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            logging.getLogger("findex").info(
                "%s: %.3f мс", function.__name__, (perf_counter() - started) * 1000
            )

    return wrapper
