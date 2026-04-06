"""Retry logic for network filesystem operations."""

import errno

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


def is_host_down(exc):
    return isinstance(exc, OSError) and exc.errno == errno.EHOSTDOWN  # 112


retry_netfs = retry(
    retry=retry_if_exception(is_host_down),   # only on EHOSTDOWN
    wait=wait_exponential(multiplier=0.1, max=5),  # 0.1s, 0.2s, 0.4s, 0.8s …
    stop=stop_after_attempt(10),
    reraise=True,                             # bubble up if it *still* fails
)
