from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from functools import wraps
from threading import BoundedSemaphore
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kotomka-llm")
_REQUEST_SLOTS = BoundedSemaphore(4)


def parallel_map(function: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """Share a bounded pool across jobs and preserve input order on collection."""
    futures = [_POOL.submit(function, item) for item in items]
    try:
        return [future.result() for future in futures]
    finally:
        for future in futures:
            future.cancel()
        # No requests from a failed stage may outlive its job's artifact access.
        wait(futures)


def limit_llm_requests(function):
    @wraps(function)
    def limited(*args, **kwargs):
        with _REQUEST_SLOTS:
            return function(*args, **kwargs)
    return limited
