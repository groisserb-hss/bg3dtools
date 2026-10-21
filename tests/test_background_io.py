"""background_io: the ordered write-behind thread and the ordered read-ahead.

The contract under test is the one a pipeline's gate files rely on: jobs run in submission order, a
failure stops the writer (nothing after it -- a gate least of all -- is written) and reaches the
submitting thread as the same exception, jobs work on a snapshot of their arguments, and the queue is
bounded.
"""
import contextvars
import errno
import logging
import threading
import time
from functools import partial

import numpy as np
import pytest

from bg3dtools.utils import background_io as bio
from bg3dtools.utils.background_io import (
    IMMEDIATE, background_io, deliver, prefetch, read_ahead, writer)


def _holding(event):
    """A job that holds the writer until *event* is set (to observe what waits behind it). A closure,
    so the event is shared, not copied: a job's ARGUMENTS are snapshots, its closure is not."""
    def hold():
        assert event.wait(5), 'test gate never opened'
    return hold


def test_jobs_run_in_submission_order():
    done = []
    with background_io(read_ahead=4) as out:
        for i in range(20):
            out.submit(partial(done.append, i))
    assert done == list(range(20))


def test_a_gate_lands_after_the_data_submitted_before_it(tmp_path):
    def write(name):
        time.sleep(0.01)
        (tmp_path / name).write_text(name)

    def gate():
        assert sorted(p.name for p in tmp_path.iterdir()) == ['a', 'b', 'c']
        (tmp_path / 'gate').write_text('ok')

    with background_io() as out:
        for name in 'abc':
            out.submit(partial(write, name))
        out.submit(gate)
    assert (tmp_path / 'gate').read_text() == 'ok'


def test_a_failure_stops_the_writer_and_reaches_the_submitter_unchanged():
    ran = []
    failure = OSError(errno.EHOSTDOWN, 'Host is down')

    def fail():
        raise failure

    with pytest.raises(OSError) as info:
        with background_io() as out:
            out.submit(partial(ran.append, 'data'))
            out.submit(fail)
            out.submit(partial(ran.append, 'gate'))
            out.wait()                              # waits; the stop surfaces at check()
            with pytest.raises(OSError) as checked:
                out.check()
            assert checked.value is failure
            out.submit(partial(ran.append, 'after the stop'))
    assert info.value is failure and info.value.errno == errno.EHOSTDOWN
    assert ran == ['data']


def test_check_raises_once_the_writer_has_stopped():
    raised = []

    def fail():
        raised.append(ValueError('bad job'))
        raise raised[-1]

    with pytest.raises(ValueError) as info:
        with background_io() as out:
            out.submit(fail)
            while out.failure is None:
                time.sleep(0.001)
            with pytest.raises(ValueError) as checked:
                out.check()
            assert checked.value is raised[0]
    assert info.value is raised[0]


def _raise(exc):
    raise exc


def test_a_non_fatal_job_only_warns(caplog):
    ran = []
    with caplog.at_level(logging.WARNING, logger=bio.__name__):
        with background_io() as out:
            out.submit(partial(_raise, OSError(errno.EACCES, 'read-only')), fatal=False, tag='video')
            out.submit(partial(ran.append, 'gate'))
    assert ran == ['gate']
    assert 'non-fatal' in caplog.text and 'video' in caplog.text


def test_outside_a_block_jobs_run_in_place_and_raise_in_place():
    assert writer() is IMMEDIATE
    ran = []
    writer().submit(partial(ran.append, 1))
    assert ran == [1]
    with pytest.raises(OSError):
        writer().submit(partial(_raise, OSError('copy failed')), fatal=False)


def test_a_job_sees_its_arguments_as_they_were_submitted():
    hold = threading.Event()
    seen = []
    data = np.zeros(3)
    with background_io() as out:
        out.submit(_holding(hold))
        out.submit(partial(seen.append, data))      # copied now
        data[:] = 7                                 # the submitter moves on and changes it
        hold.set()
    assert seen[0].tolist() == [0, 0, 0]


def test_submit_blocks_while_the_queue_is_full():
    hold = threading.Event()
    with background_io(max_pending=2) as out:
        out.submit(_holding(hold))            # taken by the thread, which then waits
        out.submit(partial(time.sleep, 0))
        out.submit(partial(time.sleep, 0))          # the queue now holds max_pending jobs
        submitter = threading.Thread(target=out.submit, args=(partial(time.sleep, 0),))
        submitter.start()
        submitter.join(0.2)
        assert submitter.is_alive(), 'submit should block on a full queue'
        hold.set()
        submitter.join(5)
        assert not submitter.is_alive()


def test_wait_for_tagged_jobs_does_not_wait_for_later_ones():
    hold = threading.Event()
    done = []
    with background_io() as out:
        out.submit(partial(done.append, 'a1'), tag=('a', 1))
        out.submit(partial(done.append, 'b1'), tag=('b', 1))
        out.submit(partial(done.append, 'a3'), tag=('a', 3))
        out.submit(_holding(hold), tag=('b', 9))
        out.wait(lambda tag: tag[0] == 'a' and tag[1] >= 2)
        assert done == ['a1', 'b1', 'a3']           # every job up to a3 ran, the held one still waits
        out.wait(lambda tag: tag[0] == 'c')          # nothing pending matches: no wait
        hold.set()


stamp = contextvars.ContextVar('stamp', default=None)


def test_jobs_run_in_the_submitters_context_with_synchronous_io():
    seen = []

    def job():
        seen.append((stamp.get(), writer() is IMMEDIATE, read_ahead()))

    with background_io(read_ahead=4) as out:
        token = stamp.set('scan A')
        out.submit(job)
        stamp.set('scan B')                          # later context changes do not reach the job
        out.submit(job)
        stamp.reset(token)
    assert seen == [('scan A', True, 0), ('scan B', True, 0)]


def test_a_synchronous_block():
    with background_io(read_ahead=0) as out:
        assert out is IMMEDIATE and writer() is IMMEDIATE and read_ahead() == 0
        assert [t is threading.current_thread()
                for t in prefetch(lambda i: threading.current_thread(), range(3))] == [True] * 3
    assert read_ahead() == bio.READ_AHEAD


def test_the_block_drains_before_its_exception_propagates():
    hold = threading.Event()
    done = []
    with pytest.raises(KeyError):
        with background_io() as out:
            out.submit(_holding(hold))
            out.submit(partial(done.append, 'queued before the failure'))
            hold.set()
            raise KeyError('the session failed')
    assert done == ['queued before the failure']


# --- prefetch ---------------------------------------------------------------------------------------

def test_prefetch_keeps_order_and_depth():
    lock = threading.Lock()
    state = {'running': 0, 'peak': 0}

    def load(i):
        with lock:
            state['running'] += 1
            state['peak'] = max(state['peak'], state['running'])
        time.sleep(0.002 * (i % 3))
        with lock:
            state['running'] -= 1
        return i * i

    assert list(prefetch(load, range(30), depth=3)) == [i * i for i in range(30)]
    assert 1 < state['peak'] <= 3


def test_prefetch_raises_at_the_failing_item():
    def load(i):
        if i == 3:
            raise FileNotFoundError(f'frame {i}')
        return i

    got = []
    with pytest.raises(FileNotFoundError, match='frame 3'):
        for x in prefetch(load, range(10), depth=4):
            got.append(x)
    assert got == [0, 1, 2]


def test_leaving_prefetch_early_cancels_the_loads_not_started():
    started = []

    def load(i):
        started.append(i)
        time.sleep(0.005)
        return i

    for x in prefetch(load, range(100), depth=2):
        if x == 4:
            break
    time.sleep(0.05)
    assert len(started) <= 4 + 1 + 2


def test_prefetch_runs_in_the_callers_context():
    token = stamp.set('scan C')
    try:
        assert list(prefetch(lambda _: (stamp.get(), read_ahead()), range(3), depth=2)) == [('scan C', 0)] * 3
    finally:
        stamp.reset(token)


# --- deliver ----------------------------------------------------------------------------------------

def test_deliver_replaces_the_target_and_removes_the_local_file(tmp_path):
    local, dst = tmp_path / 'encoded.mp4', tmp_path / 'out' / 'video.mp4'
    dst.parent.mkdir()
    dst.write_bytes(b'old')
    local.write_bytes(b'new')
    with background_io():
        deliver(local, dst)
    assert dst.read_bytes() == b'new' and not local.exists()


def test_a_failed_delivery_is_non_fatal_in_a_block_and_raises_outside(tmp_path, caplog):
    def staged():
        local = tmp_path / 'encoded.mp4'
        local.write_bytes(b'new')
        return local

    missing_dir = tmp_path / 'no-such-dir' / 'video.mp4'
    with caplog.at_level(logging.WARNING, logger=bio.__name__):
        with background_io():
            deliver(staged(), missing_dir)
    assert 'non-fatal' in caplog.text
    assert not (tmp_path / 'encoded.mp4').exists()
    with pytest.raises(FileNotFoundError):
        deliver(staged(), missing_dir)
    assert not (tmp_path / 'encoded.mp4').exists()
