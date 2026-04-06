"""Miscellaneous I/O wrappers with network filesystem retry."""

import logging

from ._retry import retry_netfs

__all__ = ["load_mat", "save_mat", "dump_json", "load_json", "read_text", "write_text"]


@retry_netfs
def load_mat(path, squeeze_me=True, struct_as_record=False):
    import scipy.io as sio

    log = logging.getLogger(__name__)
    try:
        mat = sio.loadmat(path, squeeze_me=squeeze_me, struct_as_record=struct_as_record)
    except Exception as e:
        log.error('Failed to load %s' % path)
        raise
    return mat


@retry_netfs
def save_mat(path, mdict, appendmat=True, format='5', long_field_names=False):
    import scipy.io as sio

    log = logging.getLogger(__name__)
    try:
        sio.savemat(path, mdict, appendmat=appendmat, format=format, long_field_names=long_field_names)
    except Exception as e:
        log.error('Failed to save %s' % path)
        raise


@retry_netfs
def dump_json(path, data, indent=4):
    import json

    log = logging.getLogger(__name__)
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        log.error('Failed to save %s' % path)
        raise


@retry_netfs
def load_json(path):
    import json

    log = logging.getLogger(__name__)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        log.error('Failed to load %s' % path)
        raise
    return data


@retry_netfs
def read_text(path):
    """Read a text file and return its contents as a string."""
    log = logging.getLogger(__name__)
    try:
        with open(path, 'r') as f:
            text = f.read()
    except Exception as e:
        log.error('Failed to read %s' % path)
        raise
    return text


@retry_netfs
def write_text(path, text):
    """Write a string to a text file."""
    log = logging.getLogger(__name__)
    try:
        with open(path, 'w') as f:
            f.write(text)
    except Exception as e:
        log.error('Failed to write %s' % path)
        raise
