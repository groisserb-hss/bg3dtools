"""
Utilities for suppressing output from C++ libraries.

MediaPipe and TensorFlow use C++ logging (abseil/glog) that bypasses Python's
logging system. The only reliable way to suppress these messages is to redirect
stderr at the OS file descriptor level.
"""

import os
import sys


class SuppressCppStderr:
    """
    Context manager to suppress C++ stderr output.

    Useful for suppressing verbose logs from MediaPipe, TensorFlow, and other
    C++ libraries that don't respect Python logging settings.

    Example:
        with SuppressCppStderr():
            result = landmark_video(frames)  # No C++ debug spam

    Note:
        This redirects stderr at the OS level, so it will suppress ALL stderr
        output within the context, including legitimate error messages from
        C++ code. Use judiciously.
    """

    def __enter__(self):
        self._stderr_fd = sys.stderr.fileno()
        self._old_stderr = os.dup(self._stderr_fd)
        self._devnull = open(os.devnull, 'w')
        os.dup2(self._devnull.fileno(), self._stderr_fd)
        return self

    def __exit__(self, *args):
        os.dup2(self._old_stderr, self._stderr_fd)
        os.close(self._old_stderr)
        self._devnull.close()


def suppress_cpp_stderr(func):
    """
    Decorator to suppress C++ stderr output during function execution.

    Example:
        @suppress_cpp_stderr
        def process_with_mediapipe(frames):
            return landmark_video(frames)
    """
    def wrapper(*args, **kwargs):
        with SuppressCppStderr():
            return func(*args, **kwargs)
    return wrapper
