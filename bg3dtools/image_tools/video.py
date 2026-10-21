"""
Video I/O utilities.

This module provides functions for reading and writing video files
using imageio as backend.
"""

import os
import tempfile
from collections.abc import Iterable
from typing import Iterator, List, Optional
import imageio
import numpy as np


def vreader(path: str) -> Iterator[np.ndarray]:
    """
    Read video frames as an iterator.

    A minimal replacement for skvideo.io.vreader.

    Parameters
    ----------
    path : str
        Path to the video file.

    Yields
    ------
    frame : (H, W, 3) ndarray, uint8
        RGB video frame.
    """
    # plugin : {"pyav", "ffmpeg"}, optional
    #     Backend to use.  *pyav* is fastest but needs `pip install av`.
    #     *ffmpeg* works with `pip install imageio[ffmpeg]`.

    reader = imageio.get_reader(path)  # will use ffmpeg automatically
    for frame in reader:
        yield frame


def encode_video(rgb_frames: Iterable[np.ndarray], fps: float, suffix: str = '.mp4',
                 **writer_kwargs) -> str:
    """
    Encode frames to a new LOCAL temporary file and return its path; the caller owns the file.

    Local on purpose: ffmpeg writing straight at a stale network mount blocks in kernel I/O without
    raising (imageio then waits for it forever). ``bg3dtools.utils.background_io.deliver`` moves the
    finished file to its destination in Python, where the copy is retried -- and, inside a
    ``background_io`` block, off the calling thread.

    Parameters
    ----------
    rgb_frames : iterable of (H, W, 3) uint8 ndarray
        RGB frames.
    fps : float
        Frames per second.
    suffix : str
        The temporary file's extension (it picks the container).
    **writer_kwargs
        Passed to ``imageio.get_writer`` (e.g. ``codec``, ``format``).
    """
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with imageio.get_writer(tmp_path, fps=float(fps), **writer_kwargs) as writer:
            for frame in rgb_frames:
                writer.append_data(frame)
    except BaseException:
        os.remove(tmp_path)
        raise
    return tmp_path


def save_video(
    video_path: str,
    rgb_data: Iterable[np.ndarray],
    fps: int
) -> None:
    """
    Save video frames to a file.

    Parameters
    ----------
    video_path : str
        Output video file path.
    rgb_data : iterable of (H, W, 3) ndarray
        RGB frames. Values in [0, 1] are scaled to [0, 255].
    fps : int
        Frames per second for output video.

    Notes
    -----
    Encodes to a local temporary file (``encode_video``), then delivers it to *video_path*
    (``background_io.deliver``: delete-first, retried copy) -- inside a ``background_io`` block on the
    block's writer thread, after everything the block queued before it.
    """
    from bg3dtools.utils.background_io import deliver

    rgb_data = list(rgb_data)
    if max(f.max() for f in rgb_data) <= 1.0:
        rgb_data = [255*f for f in rgb_data]
    # Expect RGB uint8; convert if needed
    frames = (fr.astype(np.uint8, copy=False) for fr in rgb_data)
    suffix = os.path.splitext(str(video_path))[1] or '.mp4'
    deliver(encode_video(frames, fps, suffix=suffix), video_path)


def load_video(video_path: str, n_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load the frames of a video file: all of them, or the first *n_frames*.

    Parameters
    ----------
    video_path : str
        Path to video file.
    n_frames : int, optional
        Read at most this many frames from the start (e.g. 1 for the frame
        size without decoding the whole video). None (default) reads all.

    Returns
    -------
    frames : list of (H, W, 3) ndarray, uint8
        RGB video frames.
    """
    if n_frames is not None and n_frames < 1:
        raise ValueError(f'load_video: n_frames must be >= 1 or None, got {n_frames}')
    reader = imageio.get_reader(str(video_path))
    try:
        # NB: do NOT use `list(reader)` — imageio's ffmpeg reader reports get_length()=inf,
        # so list() preallocates sys.maxsize (length_hint = 2**63-1) and raises MemoryError
        # regardless of free RAM. Iterating with append reads sequentially and stops at the
        # true frame count.
        frames = []
        for frame in reader:  # RGB uint8
            frames.append(frame)
            if len(frames) == n_frames:
                break
    finally:
        reader.close()
    return frames
