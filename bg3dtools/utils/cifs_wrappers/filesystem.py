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
    # copyfile, NOT copy2: copy2's copystat() sets explicit timestamps on dst,
    # which the kernel only allows for the file's owner — on CIFS mounts with
    # forceuid mapping files to another uid (e.g. the Azure Files data shares),
    # that utime raises EPERM after the data has already copied. We never need
    # source metadata on the destination (sources are typically tempfiles).
    import shutil
    log = logging.getLogger(__name__)
    try:
        shutil.copyfile(src, dst)
    except Exception as e:
        log.error('Failed to copy %s to %s: %r' % (src, dst, e))
        raise


@retry_netfs
def remove_tree(path: str):
    import shutil
    log = logging.getLogger(__name__)
    try:
        shutil.rmtree(path)
    except Exception as e:
        log.error('Failed to remove %s: %r' % (path, e))
        raise


@retry_netfs
def isfile(path):
    import os

    log = logging.getLogger(__name__)
    try:
        flag = os.path.isfile(path)
    except Exception as e:
        log.error('Failed to find %s: %r' % (path, e))
        raise
    return flag


@retry_netfs
def isdir(path):
    import os

    log = logging.getLogger(__name__)
    try:
        flag = os.path.isdir(path)
    except Exception as e:
        log.error('Failed to find %s: %r' % (path, e))
        raise
    return flag


@retry_netfs
def makedirs(path, exist_ok=True):
    import os

    log = logging.getLogger(__name__)
    try:
        os.makedirs(path, exist_ok=exist_ok)
    except Exception as e:
        log.error('Failed to make %s: %r' % (path, e))
        raise


@retry_netfs
def rename(src, dst):
    import os

    log = logging.getLogger(__name__)
    try:
        os.rename(src, dst)
    except Exception as e:
        log.error('Failed to rename %s to %s: %r' % (src, dst, e))
        raise


@retry_netfs
def remove(path):
    import os

    log = logging.getLogger(__name__)
    try:
        os.remove(path)
    except Exception as e:
        log.error('Failed to remove %s: %r' % (path, e))
        raise


delete_file = remove


@retry_netfs
def listdir(path):
    import os

    log = logging.getLogger(__name__)
    try:
        listed = os.listdir(path)
    except Exception as e:
        log.error('Failed to find %s: %r' % (path, e))
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
        log.error('Failed to glob %s/%s: %r' % (directory, pattern, e))
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
        log.error('Failed to get size of %s: %r' % (path, e))
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
        log.error('Failed to touch %s: %r' % (path, e))
        raise
