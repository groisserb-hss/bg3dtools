"""
Filesystem-based mutex utilities.

This module provides simple file-based locking for process synchronization.
"""

import os
import errno
import time
import logging

log = logging.getLogger(__name__)


def acquire_mutex(
    path: str,
    retries: int = 3,
    delay: float = 0.5
) -> bool:
    """
    Acquire a filesystem-based mutex lock.

    Creates an exclusive lock file. Only succeeds if the file doesn't exist.

    Parameters
    ----------
    path : str
        Path for the lock file.
    retries : int, optional
        Number of retries on EINTR. Default is 3.
    delay : float, optional
        Delay between retries in seconds. Default is 0.5.

    Returns
    -------
    acquired : bool
        True if lock acquired, False if already held by another process.
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags)      # atomic: succeeds only if file did not exist
        os.write(fd, b'%d\n' % os.getpid())      # optional: record owner’s PID
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except OSError as e:
        if e.errno == errno.EINTR and retries:
            time.sleep(delay)
            return acquire_mutex(path, retries-1, delay)
        raise                 # propagate anything unexpected


def release_mutex(path: str) -> None:
    """
    Release a filesystem-based mutex lock.

    Parameters
    ----------
    path : str
        Path to the lock file to remove.
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        pass