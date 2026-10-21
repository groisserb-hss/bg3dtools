"""Background I/O for work whose data lives on a slow network drive.

On a CIFS/SMB data drive every open, read and write is a network round trip, and one mount connection
gives only a few operations of useful parallelism (~4-5 on the HSS hosts' Azure Files shares; 150
concurrent reads congested it). A compute loop that reads and writes inline waits on the drive. Two
tools take that I/O off the computing thread with a handful of threads -- threads, not processes: a
thread blocked on the drive releases the GIL, and more processes would not raise one connection's
ceiling.

``background_io(read_ahead)``
    A block whose writes run on ONE background thread in the order they were submitted
    (``writer().submit``), and whose ``prefetch`` reads run up to *read_ahead* items ahead. Outside such a
    block ``writer()`` runs every job at once on the calling thread, so code that hands its writes to
    ``writer()`` behaves exactly as before for every caller that does not open a block;
    ``read_ahead=0`` makes the block synchronous too.

``prefetch(load, items)``
    ``load(item)`` for each item, in order, with up to ``read_ahead()`` loads in flight.

The writer's contract -- what makes write-behind safe for outputs that are committed by a gate file
written after them:

- ORDER. Jobs run one at a time, first in, first out: a job starts after every job submitted before it
  has finished. A gate submitted after its data therefore lands after its data, and after everything
  submitted before it.
- SNAPSHOT. ``submit`` deep-copies the job (a ``functools.partial`` with its data bound as arguments,
  or a bound method with its instance) when it is submitted: nothing the submitting thread changes
  later reaches a queued write. A closure's captured variables are NOT copied -- bind data as
  arguments.
- A FAILURE STOPS IT. A job that raises stops the writer: every job after it is dropped (a gate
  among them is never written), and the exception -- the same object, its type intact -- is raised on
  the submitting thread by its next ``check()``, or when the block exits. A job
  submitted with ``fatal=False`` (a courtesy output: a video, an auxiliary record) is logged and passed
  over instead; outside a block it raises in place, as the call it replaces did.
- BOUNDED. At most *max_pending* jobs wait; ``submit`` blocks beyond that. Memory and the lag stay
  bounded, and a writer stuck on a hung drive stalls the submitting thread as a synchronous write
  would have.
- CONTEXT. A job runs in a copy of the submitter's ``contextvars`` context (log stamps kept there
  follow it), in which the writer is the immediate one and reads are serial: whatever a job writes,
  it writes in place, in its turn.
- NO PROCESSES. A job -- and a ``prefetch`` load -- must not start a process (subprocess,
  multiprocessing, an ffmpeg behind imageio). Where subprocess forks (macOS), the fork runs
  OpenBLAS's fork handler, which stops BLAS's thread pool under any threaded BLAS call another
  thread is in, and that call never returns. Encoders therefore run on the calling thread, and only
  their finished file is queued (``deliver``, a Python copy).

Jobs carry an optional *tag* (any value the caller chooses) so a caller can ``wait`` for the jobs
that touch one place -- e.g. before deleting files a queued job may still write. ``wait`` only waits;
``check`` is where a stop surfaces, so the caller decides where that is.
"""
import contextvars
import copy
import itertools
import logging
import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial

__all__ = [
    "READ_AHEAD",
    "MAX_PENDING",
    "background_io",
    "writer",
    "read_ahead",
    "prefetch",
    "deliver",
    "OutputWriter",
    "ImmediateWriter",
]

log = logging.getLogger(__name__)

READ_AHEAD = 4      # loads in flight outside a block (one SMB connection serves ~4-5 operations at once)
MAX_PENDING = 128   # jobs a background writer holds before submit() blocks


class ImmediateWriter:
    """The writer outside a background block: every job runs at once, on the calling thread."""

    def submit(self, job, *, tag=None, fatal=True):
        job()

    def wait(self, matching=None):
        pass

    def check(self):
        pass


IMMEDIATE = ImmediateWriter()
_writer = contextvars.ContextVar('bg3dtools_background_writer', default=IMMEDIATE)
_read_ahead = contextvars.ContextVar('bg3dtools_read_ahead', default=READ_AHEAD)


def writer():
    """The current block's background writer, or the immediate one outside a block."""
    return _writer.get()


def read_ahead():
    """How many loads ``prefetch`` keeps in flight here (0: it reads serially)."""
    return _read_ahead.get()


def _serial():
    """Make the current context synchronous: immediate writes, serial reads (run inside a context copy)."""
    _writer.set(IMMEDIATE)
    _read_ahead.set(0)


def _serial_context():
    context = contextvars.copy_context()
    context.run(_serial)
    return context


class OutputWriter:
    """One thread that runs the submitted jobs in submission order (the contract: module docstring).

    Submit from one thread -- the block's. Other threads see the immediate writer through ``writer()``.
    """

    def __init__(self, max_pending=MAX_PENDING):
        self._queue = queue.Queue(maxsize=max_pending)
        self._cond = threading.Condition()
        self._submitted = 0     # jobs submitted: they are numbered 1..n
        self._finished = 0      # jobs run, passed over or dropped -- in order, so 1..m are done
        self._pending = {}      # number -> tag of every job not yet finished
        self._failure = None    # the exception that stopped the writer
        self._closed = False
        self._thread = threading.Thread(target=self._run, name='bg3dtools-writer', daemon=True)
        self._thread.start()

    @property
    def failure(self):
        """The exception that stopped the writer, or None."""
        return self._failure

    def submit(self, job, *, tag=None, fatal=True):
        """Queue *job* (a callable taking no arguments) behind every job submitted before it.

        The job is deep-copied now. A stopped writer drops it: the stop is raised by the next
        ``check()``. Blocks while *max_pending* jobs wait.
        """
        if self._closed:
            raise RuntimeError('submit() on a closed writer')
        job = copy.deepcopy(job)
        context = _serial_context()
        with self._cond:
            if self._failure is not None:
                return
            self._submitted += 1
            number = self._submitted
            self._pending[number] = tag
        self._queue.put((number, job, tag, fatal, context))

    def check(self):
        """Raise the exception that stopped the writer, if one did."""
        if self._failure is not None:
            raise self._failure

    def wait(self, matching=None):
        """Block until every job submitted so far has finished -- or, given *matching* (a
        predicate on tags), every pending job whose tag satisfies it (and so every job before
        those). A stopped writer's dropped jobs count as finished; the stop itself is ``check()``'s
        to raise."""
        with self._cond:
            if matching is None:
                last = self._submitted
            else:
                last = max((n for n, tag in self._pending.items()
                            if tag is not None and matching(tag)), default=0)
            self._cond.wait_for(lambda: self._finished >= last)

    def close(self):
        """Run (or, after a stop, drop) every queued job, then end the thread. Never raises the
        stop: ``check()`` does."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            number, job, tag, fatal, context = item
            if self._failure is None:
                try:
                    context.run(job)
                except BaseException as exc:  # noqa: BLE001 -- every failure is the submitter's to handle
                    if fatal:
                        log.error('background write failed%s -- the writer stops: %r',
                                  _where(tag), exc)
                        with self._cond:
                            self._failure = exc
                    else:
                        log.warning('background write failed (non-fatal)%s: %r', _where(tag), exc)
            with self._cond:
                del self._pending[number]
                self._finished = number
                self._cond.notify_all()


def _where(tag):
    return '' if tag is None else f' [{tag}]'


@contextmanager
def background_io(read_ahead=READ_AHEAD, max_pending=MAX_PENDING):
    """A block whose writes run on one background thread in order, and whose ``prefetch`` reads
    keep *read_ahead* loads in flight. ``read_ahead=0``: everything in the block is synchronous.

    Yields the writer. On exit the writer runs every queued job and its thread ends. A clean exit
    then raises the writer's stop, if any; when the block is already raising, the queued jobs are
    still run (they were submitted before the failure: a synchronous run would have written them),
    a stop of the writer's own is logged, and the block's exception propagates.
    """
    if read_ahead <= 0:
        out = IMMEDIATE
        tokens = (_writer.set(out), _read_ahead.set(0))
    else:
        out = OutputWriter(max_pending)
        tokens = (_writer.set(out), _read_ahead.set(int(read_ahead)))
    try:
        yield out
    except BaseException:
        if out is not IMMEDIATE:
            out.close()
            if out.failure is not None:
                log.error('the background writer had stopped too: %r', out.failure)
        raise
    else:
        if out is not IMMEDIATE:
            out.close()
            out.check()
    finally:
        _writer.reset(tokens[0])
        _read_ahead.reset(tokens[1])


def prefetch(load, items, depth=None):
    """``load(item)`` for each of *items*, in order, with up to *depth* (default ``read_ahead()``)
    loads in flight on worker threads; ``depth`` 0 loads serially on the calling thread.

    For reads only: a load must not write, nor start a process (module docstring). Loads run in a
    copy of the caller's context in which writes are immediate and nested reads serial. A load's exception is raised when its item is
    reached, after the items before it were yielded. Leaving the loop early (break, an exception,
    closing the generator) cancels the loads not yet started and waits for those running.
    """
    depth = read_ahead() if depth is None else int(depth)
    if depth <= 0:
        for item in items:
            yield load(item)
        return
    context = _serial_context()
    items = iter(items)
    with ThreadPoolExecutor(max_workers=depth, thread_name_prefix='bg3dtools-read') as pool:
        in_flight = deque(pool.submit(context.copy().run, load, item)
                          for item in itertools.islice(items, depth))
        try:
            while in_flight:
                result = in_flight.popleft().result()
                for item in itertools.islice(items, 1):
                    in_flight.append(pool.submit(context.copy().run, load, item))
                yield result
        finally:
            for future in in_flight:
                future.cancel()


def deliver(local_path, dst):
    """Move the finished local file *local_path* to *dst* (typically on the network drive): delete
    *dst*, copy, remove the local file -- queued on the writer as a courtesy output (``fatal=False``).

    The house pattern for files a subprocess writes (ffmpeg): encode to local disk, where a stale
    mount cannot block the encoder, and keep the network I/O in Python, where ``copy_file`` retries
    it. Delete first: copyfile overwrite-opens its destination, and an overwrite-open of a live file
    on a bad SMB session can destroy the target before failing -- a vacant name is the safe landing.
    """
    writer().submit(partial(_move_to, str(local_path), str(dst)), fatal=False)


def _move_to(local_path, dst):
    from .cifs_wrappers.filesystem import copy_file
    try:
        try:
            os.remove(dst)
        except FileNotFoundError:
            pass                # the name being gone is the goal
        copy_file(local_path, dst)
    finally:
        try:
            os.remove(local_path)
        except FileNotFoundError:
            pass
