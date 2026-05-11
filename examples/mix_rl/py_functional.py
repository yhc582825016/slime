"""Small utilities copied from Reasoning360/verl for standalone rewards."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import queue
import signal
from functools import wraps
from typing import Any, Callable


def _mp_target_wrapper(target_func: Callable, mp_queue: multiprocessing.Queue, args: tuple, kwargs: dict[str, Any]):
    try:
        result = target_func(*args, **kwargs)
        mp_queue.put((True, result))
    except Exception as exc:
        try:
            pickle.dumps(exc)
            mp_queue.put((False, exc))
        except (pickle.PicklingError, TypeError):
            mp_queue.put((False, RuntimeError(f"Original exception type {type(exc).__name__} not pickleable: {exc}")))


def timeout_limit(seconds: float, use_signals: bool = False):
    def decorator(func):
        if use_signals:
            if os.name != "posix":
                raise NotImplementedError(f"Unsupported OS: {os.name}")

            @wraps(func)
            def wrapper_signal(*args, **kwargs):
                def handler(signum, frame):
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds.")

                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.setitimer(signal.ITIMER_REAL, seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper_signal

        @wraps(func)
        def wrapper_mp(*args, **kwargs):
            q = multiprocessing.Queue(maxsize=1)
            process = multiprocessing.Process(target=_mp_target_wrapper, args=(func, q, args, kwargs))
            process.start()
            process.join(timeout=seconds)

            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds.")

            try:
                success, result_or_exc = q.get(timeout=0.1)
                if success:
                    return result_or_exc
                raise result_or_exc
            except queue.Empty as err:
                raise TimeoutError(f"Operation timed out or finished unexpectedly, exitcode={process.exitcode}.") from err
            finally:
                q.close()
                q.join_thread()

        return wrapper_mp

    return decorator

