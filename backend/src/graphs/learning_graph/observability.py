import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def timed(name: str):
    """
    Wrap a graph node to log its wall-clock duration on every run.

    Logs node completion at INFO and a warning with the elapsed time on error
    before re-raising, so footprint regressions are visible in the logs.

    :param name: Node name to attach to the log lines.
    :type name: str
    :return: Decorator wrapping the node callable with timing.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    "node=%s error=%s elapsed_ms=%.1f",
                    name,
                    exc,
                    (time.perf_counter() - started) * 1e3,
                )
                raise
            logger.info(
                "node=%s elapsed_ms=%.1f",
                name,
                (time.perf_counter() - started) * 1e3,
            )
            return result

        return wrapper

    return decorator