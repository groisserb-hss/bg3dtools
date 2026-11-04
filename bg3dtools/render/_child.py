"""Run one render call in a fresh Python process: ``run_in_child(fn, *args)`` in the parent,
``python -m bg3dtools.render._child JOB RESULT`` in the child.

Why a fresh process per call: Open3D's headless ``OffscreenRenderer`` does not survive many renderers in
one process. On SAR-PC (open3d 0.19, EGL headless, 2026-11-04) a process segfaulted in ``add_geometry``
once it had built four or five renderers, and a seated registration run that built one per video went on
silently returning black frames after about a hundred; one renderer in a fresh process rendered every frame
(280 of 280) every time. Every renderer this package builds therefore lives in a child that exits after
its one call (``scan._in_render_child``), and a native crash there costs that call, not the caller.

A plain ``subprocess`` rather than ``multiprocessing``: the child's import order is this module's, not a
re-import of the caller's ``__main__`` (which ``spawn`` does), and it matters. On SAR-PC the renderer
segfaulted on its first frame in a process that had imported only ``bg3dtools.render.scan`` before
rendering, and rendered cleanly in one that had imported ``bg3dtools.mesh`` (open3d, torch, igl, scipy)
first, which is why the child imports it before anything renders. Which library's load order is the
trigger was not isolated.

The job and the result cross as pickle files in a private temporary directory (a frame list can be
hundreds of MB, more than a pipe should buffer). The child sends back the call's value, or its
exception, plus the WARNING-and-above records it logged, which the parent re-emits under their own
logger names so they reach the caller's log handlers. A child that dies before writing its result
raises ``RenderChildDied`` naming its exit code (-11: a segfault).
"""
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


class RenderChildDied(RuntimeError):
    """The render child exited without a result: a native crash (exit -11 is a segfault) or a kill."""


def run_in_child(fn, *args):
    """``fn(*args)`` in a fresh Python process; its value, or its exception re-raised here.

    *fn* is pickled by reference (a module-level function) and *args* by value. The child inherits the
    environment, so a ``EGL_PLATFORM`` the render probe adopted applies there too. Blocks until the child
    exits; its stdout and stderr are this process's (Open3D's native start-up lines print there).
    """
    with tempfile.TemporaryDirectory(prefix='bg3dtools-render-') as tmp:
        job, result = Path(tmp, 'job.pkl'), Path(tmp, 'result.pkl')
        job.write_bytes(pickle.dumps((fn, args), protocol=pickle.HIGHEST_PROTOCOL))
        proc = subprocess.run([sys.executable, '-m', __name__, str(job), str(result)])
        if not result.is_file():
            raise RenderChildDied(f'{getattr(fn, "__name__", fn)}: the render child exited with code '
                                  f'{proc.returncode} and no result')
        ok, value, records = pickle.loads(result.read_bytes())
    for name, level, message in records:
        logging.getLogger(name).log(level, '%s', message)
    if not ok:
        raise value
    return value


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append((record.name, record.levelno, record.getMessage()))


def _main(job, result):
    from bg3dtools.render.scan import _offscreen_crash_probe_ok
    _offscreen_crash_probe_ok()          # first: an 'ok-surfaceless' host needs EGL_PLATFORM set before open3d loads
    import bg3dtools.mesh  # noqa: F401  (loads open3d; before anything renders: see the module docstring)

    collect = _Collect()
    logging.getLogger().addHandler(collect)
    fn, args = pickle.loads(Path(job).read_bytes())
    try:
        outcome = (True, fn(*args))
    except Exception as exc:  # noqa: BLE001 -- the parent re-raises it
        outcome = (False, exc)
    try:
        payload = pickle.dumps(outcome + (collect.records,), protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # noqa: BLE001 -- an unpicklable value or exception still reaches the parent
        payload = pickle.dumps((False, RuntimeError(f'render child: {outcome[1]!r} ({exc})'),
                                collect.records), protocol=pickle.HIGHEST_PROTOCOL)
    staged = Path(f'{result}.partial')
    staged.write_bytes(payload)
    os.replace(staged, result)            # the parent sees a whole result or none


if __name__ == '__main__':
    _main(*sys.argv[1:])
