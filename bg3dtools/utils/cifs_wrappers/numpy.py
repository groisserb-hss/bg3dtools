"""NumPy I/O wrappers with network filesystem retry."""

import logging

from ._retry import retry_netfs

__all__ = ["save_np", "save_npz", "load_np", "save_csv", "load_csv"]


@retry_netfs
def save_np(path: str, array):
    import numpy as np
    log = logging.getLogger(__name__)
    try:
        np.save(path, array)
    except Exception as e:
        log.error('Failed to save %s' % path)
        raise


@retry_netfs
def save_npz(file_path, *args, **kwargs):
    import numpy as np
    log = logging.getLogger(__name__)
    try:
        np.savez_compressed(file_path, *args, **kwargs)
    except Exception as e:
        log.error('Failed to save %s' % file_path)
        raise


@retry_netfs
def load_np(path, allow_pickle=True):
    import numpy as np
    log = logging.getLogger(__name__)
    try:
        data = np.load(path, allow_pickle=allow_pickle)
    except Exception as e:
        log.error('Failed to load %s' % path)
        raise
    return data


@retry_netfs
def save_csv(path, data, delimiter=',', fmt='%s'):
    import numpy as np

    log = logging.getLogger(__name__)
    try:
        np.savetxt(path, data, delimiter=delimiter, fmt=fmt)
    except Exception as e:
        log.error('Failed to save %s' % path)
        raise


@retry_netfs
def load_csv(path, delimiter=','):
    import numpy as np

    log = logging.getLogger(__name__)
    try:
        csv = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
    except Exception as e:
        log.error('Failed to load %s' % path)
        raise
    return csv
