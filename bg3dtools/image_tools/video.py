"""
Video I/O utilities.

This module provides functions for reading and writing video files
using imageio as backend.
"""

from collections.abc import Iterable
from typing import Iterator, List
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
    """
    writer = imageio.get_writer(str(video_path), fps=float(fps))
    try:

        rgb_data = list(rgb_data)
        if max(f.max() for f in rgb_data) <= 1.0:
            rgb_data = [255*f for f in rgb_data]

        for fr in rgb_data:
            # Expect RGB uint8; convert if needed
            fr_u8 = fr.astype(np.uint8, copy=False)
            writer.append_data(fr_u8)
    finally:
        writer.close()


def load_video(video_path: str) -> List[np.ndarray]:
    """
    Load all frames from a video file.

    Parameters
    ----------
    video_path : str
        Path to video file.

    Returns
    -------
    frames : list of (H, W, 3) ndarray, uint8
        RGB video frames.
    """
    reader = imageio.get_reader(str(video_path))
    try:
        frames = list(reader)  # RGB uint8
    finally:
        reader.close()
    return frames
