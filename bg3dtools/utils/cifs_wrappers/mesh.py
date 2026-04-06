"""Mesh I/O wrappers with network filesystem retry."""

import logging

from ._retry import retry_netfs

__all__ = ["read_mesh", "read_plydata", "write_plydata"]


@retry_netfs
def read_mesh(path, **kwargs):
    """Read a triangle mesh via bg3dtools.mesh.mesh_io.read_triangle_mesh."""
    from bg3dtools.mesh.mesh_io import read_triangle_mesh

    log = logging.getLogger(__name__)
    try:
        result = read_triangle_mesh(str(path), **kwargs)
    except Exception as e:
        log.error('Failed to read mesh %s' % path)
        raise
    return result


@retry_netfs
def read_plydata(path):
    from plyfile import PlyData

    log = logging.getLogger(__name__)
    try:
        with open(path, 'rb') as f:
            plydata = PlyData.read(f)
    except Exception as e:
        log.error('Failed to load %s' % path)
        raise
    return plydata


@retry_netfs
def write_plydata(path, plydata):

    log = logging.getLogger(__name__)
    try:
        plydata.write(path)
    except Exception as e:
        log.error('Failed to save %s' % path)
        raise
