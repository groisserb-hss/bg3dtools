"""
Network filesystem wrappers with automatic retry.

This package provides wrappers around common file operations that automatically
retry on network filesystem errors (EHOSTDOWN). Useful for operations on
CIFS/SMB mounted drives that may experience intermittent connectivity issues.

All functions use exponential backoff retry logic for robustness.
"""

from ._retry import is_host_down, retry_netfs
from .numpy import save_np, save_npz, load_np, save_csv, load_csv
from .image import load_image, image_dims, save_image, save_video
from .mesh import read_mesh, read_plydata, write_plydata
from .filesystem import (
    copy_file,
    delete_file,
    remove_tree,
    isfile,
    isdir,
    makedirs,
    rename,
    remove,
    listdir,
    glob_path,
    getsize,
    touch,
)
from .misc import load_mat, save_mat, dump_json, load_json, read_text, write_text

__all__ = [
    # retry
    "is_host_down",
    "retry_netfs",
    # numpy
    "save_np",
    "save_npz",
    "load_np",
    "save_csv",
    "load_csv",
    # image
    "load_image",
    "image_dims",
    "save_image",
    "save_video",
    # mesh
    "read_mesh",
    "read_plydata",
    "write_plydata",
    # filesystem
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
    # misc
    "load_mat",
    "save_mat",
    "dump_json",
    "load_json",
    "read_text",
    "write_text",
]
