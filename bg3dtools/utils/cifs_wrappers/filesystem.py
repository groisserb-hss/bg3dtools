"""Filesystem operation wrappers with network filesystem retry."""

import logging

from ._retry import retry_netfs

__all__ = [
    "copy_file",
    "delete_file",
    "remove_tree",
    "isfile",
    "isdir",
    "makedirs",
    "rename",
    "remove",
    "listdir",
    "glob_path",
    "getsize",
    "touch",
]


@retry_netfs
def copy_file(src: str, dst: str):
    import shutil
    log = logging.getLogger(__name__)
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        log.error('Failed to copy %s to %s' % (src, dst))
        raise


@retry_netfs
def remove_tree(path: str):
    import shutil
    log = logging.getLogger(__name__)
    try:
        shutil.rmtree(path)
    except Exception as e:
        log.error('Failed to remove %s' % path)
        raise


@retry_netfs
def isfile(path):
    import os

    log = logging.getLogger(__name__)
    try:
        flag = os.path.isfile(path)
    except Exception as e:
        log.error('Failed to find %s' % path)
        raise
    return flag


@retry_netfs
def isdir(path):
    import os

    log = logging.getLogger(__name__)
    try:
        flag = os.path.isdir(path)
    except Exception as e:
        log.error('Failed to find %s' % path)
        raise
    return flag


@retry_netfs
def makedirs(path, exist_ok=True):
    import os

    log = logging.getLogger(__name__)
    try:
        os.makedirs(path, exist_ok=exist_ok)
    except Exception as e:
        log.error('Failed to make %s' % path)
        raise


@retry_netfs
def rename(src, dst):
    import os

    log = logging.getLogger(__name__)
    try:
        os.rename(src, dst)
    except Exception as e:
        log.error('Failed to rename %s to %s' % (src, dst))
        raise


@retry_netfs
def remove(path):
    import os

    log = logging.getLogger(__name__)
    try:
        os.remove(path)
    except Exception as e:
        log.error('Failed to remove %s' % path)
        raise


delete_file = remove


@retry_netfs
def listdir(path):
    import os

    log = logging.getLogger(__name__)
    try:
        listed = os.listdir(path)
    except Exception as e:
        log.error('Failed to find %s' % path)
        raise
    return listed


@retry_netfs
def glob_path(directory, pattern):
    """List Path objects matching a glob pattern in a directory."""
    from pathlib import Path

    log = logging.getLogger(__name__)
    try:
        result = sorted(Path(directory).glob(pattern))
    except Exception as e:
        log.error('Failed to glob %s/%s' % (directory, pattern))
        raise
    return result


@retry_netfs
def getsize(path):
    """Return the size of a file in bytes."""
    import os

    log = logging.getLogger(__name__)
    try:
        size = os.path.getsize(path)
    except Exception as e:
        log.error('Failed to get size of %s' % path)
        raise
    return size


@retry_netfs
def touch(path):
    """Create an empty file (or update mtime)."""
    log = logging.getLogger(__name__)
    try:
        with open(path, 'a') as f:
            pass
    except Exception as e:
        log.error('Failed to touch %s' % path)
        raise
