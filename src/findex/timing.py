"""Декоратор вимірювання часу зі збереженням метаданих функції."""

import logging
from collections.abc import Callable
from functools import wraps
from time import perf_counter


def timed[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            logging.getLogger("findex").debug(
                "%s: %.3f мс", function.__name__, (perf_counter() - started) * 1000
            )

    return wrapper
