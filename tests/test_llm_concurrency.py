from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
import time

from kotomka.providers.llm.concurrency import limit_llm_requests, parallel_map


def test_transport_budget_is_shared_by_multiple_callers():
    start = Barrier(9)
    occupied = Event()
    release = Event()
    lock = Lock()
    active = peak = 0

    @limit_llm_requests
    def request(index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 4:
                occupied.set()
        try:
            assert release.wait(timeout=3)
            return index
        finally:
            with lock:
                active -= 1

    def caller(index):
        start.wait(timeout=3)
        return request(index)

    with ThreadPoolExecutor(max_workers=8) as callers:
        futures = [callers.submit(caller, index) for index in range(8)]
        start.wait(timeout=3)
        try:
            assert occupied.wait(timeout=3)
            time.sleep(0.05)
            assert peak == 4
        finally:
            release.set()
        assert [future.result() for future in futures] == list(range(8))


def test_multiple_jobs_share_the_same_batch_pool():
    ready = Barrier(4)
    def task(index):
        ready.wait(timeout=3)
        return index
    with ThreadPoolExecutor(max_workers=2) as jobs:
        futures = [jobs.submit(parallel_map, task, list(range(4))) for _ in range(2)]
        assert [future.result() for future in futures] == [list(range(4)), list(range(4))]
