"""
Scheduling and timing utilities.

This module provides classes for timing code execution, tracking
running averages, monitoring convergence, and suppressing output.
"""

from __future__ import print_function
from os.path import join, isfile, getsize
from time import sleep, time
from typing import List, Optional
import sys
import os
import numpy as np


class AverageMeter:
    """
    Compute and store running average and current value.

    Attributes
    ----------
    val : float
        Most recent value.
    avg : float
        Running average of all values.
    sum : float
        Sum of all values.
    count : int
        Number of values added.

    Examples
    --------
    >>> meter = AverageMeter()
    >>> meter.update(10)
    >>> meter.update(20)
    >>> meter.avg
    15.0
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset all statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Add a new value to the running average.

        Parameters
        ----------
        val : float
            Value to add.
        n : int, optional
            Weight/count for this value. Default is 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """
    Simple timer for measuring code execution time.

    Supports tic/toc interface with cumulative timing statistics.

    Attributes
    ----------
    total_time : float
        Cumulative time across all toc() calls.
    calls : int
        Number of toc() calls.
    diff : float
        Time elapsed in most recent tic/toc interval.
    average_time : float
        Average time per call.

    Examples
    --------
    >>> timer = Timer(tic=True)
    >>> # ... do work ...
    >>> elapsed = timer.toc()
    """

    def __init__(self, tic: bool = False):
        self.total_time = 0.
        self.calls = 0
        self.start_time = time() if tic else 0.
        self.diff = 0.
        self.average_time = 0.

    def reset(self):
        """Reset all timing statistics."""
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.average_time = 0

    def tic(self):
        """Start the timer."""
        self.start_time = time()

    def toc(self, average: bool = False, update: bool = True) -> float:
        """
        Stop the timer and return elapsed time.

        Parameters
        ----------
        average : bool, optional
            If True, return average time. Default is False.
        update : bool, optional
            If True, update cumulative statistics. Default is True.

        Returns
        -------
        elapsed : float
            Elapsed time (or average if requested).
        """
        self.diff = time() - self.start_time
        if update:
            self.total_time += self.diff
            self.calls += 1
        self.average_time = self.total_time / self.calls if self.calls > 0 else -1
        if average:
            return self.average_time
        else:
            return self.diff

    def toctic(self, average: bool = False) -> float:
        """
        Stop timer, record time, and immediately restart.

        Parameters
        ----------
        average : bool, optional
            If True, return average time. Default is False.

        Returns
        -------
        elapsed : float
            Elapsed time from previous tic.
        """
        tocval = self.toc(average)
        self.start_time = time()
        return tocval


class ConvergenceScheduler:
    """
    Monitor convergence of an iterative process.

    Tracks a metric over iterations and determines when convergence
    is reached based on relative change within a sliding window.

    Parameters
    ----------
    thresh : float, optional
        Convergence threshold for relative change. Default is 0.02.
    window : int, optional
        Window size for averaging. Default is 3.
    max_iter : int, optional
        Maximum iterations before forced completion. Default is None (no limit).

    Attributes
    ----------
    steps : int
        Number of values pushed.
    current : float
        Most recent value.
    complete : bool
        Whether convergence criteria are met.
    """

    def __init__(
        self,
        thresh: float = 0.02,
        window: int = 3,
        max_iter: Optional[int] = None
    ):
        self._thresh = thresh
        self._window = window
        self._max_iter = np.inf if max_iter is None else max_iter
        self._history = []

    @property
    def steps(self) -> int:
        return len(self._history)

    @property
    def current(self) -> float:
        return self._history[-1]

    def ratio(self) -> float:
        w = self._window
        old_avg = np.mean(self._history[-2*w:-w])
        new_avg = np.mean(self._history[-w:])
        return np.abs(old_avg - new_avg) / np.abs(new_avg)

    @property
    def complete(self) -> bool:
        if self.steps > self._max_iter:
            return True

        if self.steps < 2 * self._window:
            return False

        return self.ratio() < self._thresh

    def push(self, new_val: float):
        """
        Add a new value to the history.

        Parameters
        ----------
        new_val : float
            New metric value to track.
        """
        self._history.append(new_val)


def wait_for_completion(
    folder: str,
    filenames: List[str],
    process: str = 'process'
) -> None:
    """
    Block until all specified files exist and are non-empty.

    Parameters
    ----------
    folder : str
        Directory containing the files.
    filenames : list of str
        File names to wait for.
    process : str, optional
        Process name for status messages. Default is 'process'.
    """
    print('* Waiting for %s to complete' % process)

    all_finished = False
    i = 1
    while not all_finished:
        all_finished = True
        for fname in filenames:
            filepath = join(folder, fname)
            all_finished = all_finished and isfile(filepath) and getsize(filepath) > 0
        if not all_finished:
            print('.', end='')
            if i % 70 == 0:
                print('.', flush=True)
            sleep(5.0)
            i += 1
    print('complete', flush=True)


class HiddenPrints:
    """
    Context manager to suppress stdout output.

    Examples
    --------
    >>> with HiddenPrints():
    ...     print("This won't be shown")
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout