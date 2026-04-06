"""
Filesystem utilities.

This module provides helper functions for file compression and manipulation.
"""

from typing import Optional
import gzip
import shutil
import os


def zip_file(
    infile_name: str,
    outfile_name: Optional[str] = None,
    delete_orig: bool = True
) -> None:
    """
    Compress a file using gzip.

    Parameters
    ----------
    infile_name : str
        Path to input file.
    outfile_name : str, optional
        Path for compressed output. Default is infile_name + '.gz'.
    delete_orig : bool, optional
        If True, delete original file after compression. Default is True.
    """
    if outfile_name is None:
        outfile_name = infile_name + '.gz'

    with open(infile_name, 'rb') as f_in:
        with gzip.open(outfile_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if outfile_name != infile_name and delete_orig:
        os.remove(infile_name)
