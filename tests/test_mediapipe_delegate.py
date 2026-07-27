"""Tests for the MediaPipe GPU/CPU delegate selection.

mediapipe is an optional dep (the ``vision`` extra) and is not installed
everywhere, so these tests stub ``_delegate._delegates`` -- the one seam that
touches mediapipe -- with plain sentinels. That keeps the fallback logic under
test on every host, which matters because the real GPU path is exactly the one
that cannot be exercised on a machine without a working GL stack.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Import the module directly by path: bg3dtools.pose_landmarking.__init__ pulls in
# mediapipe/cv2, which need not be installed for these tests to be meaningful.
_PATH = Path(__file__).resolve().parents[1] / 'bg3dtools' / 'pose_landmarking' / '_delegate.py'
_spec = importlib.util.spec_from_file_location('_bg3d_delegate_under_test', _PATH)
delegate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = delegate
_spec.loader.exec_module(delegate)

GPU, CPU = 'GPU-DELEGATE', 'CPU-DELEGATE'


@pytest.fixture(autouse=True)
def _stub_delegates(monkeypatch):
    """Replace the only mediapipe-touching function, and reset process caches."""
    monkeypatch.setattr(delegate, '_delegates', lambda: (GPU, CPU))
    delegate._reset_caches()
    monkeypatch.delenv('MEDIAPIPE_USE_GPU', raising=False)
    yield
    delegate._reset_caches()


def _recorder(fail_on=()):
    """A fake `build`: records the delegates it was asked for, fails on some."""
    seen = []

    def build(d):
        seen.append(d)
        if d in fail_on:
            raise RuntimeError('no GL context')
        return 'detector(%s)' % d

    build.seen = seen
    return build


# ---------------------------------------------------------------------------
# gpu_requested
# ---------------------------------------------------------------------------

def test_gpu_is_opt_in_by_default():
    """The headline behaviour: no env, no argument -> CPU. Every caller that
    existed before the delegate module keeps its original execution path."""
    assert delegate.gpu_requested() is False


@pytest.mark.parametrize('value', ['1', 'true', 'TRUE', 'yes', 'on', '  1  ', 'On'])
def test_env_var_opts_in(monkeypatch, value):
    monkeypatch.setenv('MEDIAPIPE_USE_GPU', value)
    assert delegate.gpu_requested() is True


@pytest.mark.parametrize('value', ['0', 'false', 'no', 'off', '', 'maybe', '2'])
def test_env_var_anything_else_stays_on_cpu(monkeypatch, value):
    monkeypatch.setenv('MEDIAPIPE_USE_GPU', value)
    assert delegate.gpu_requested() is False


@pytest.mark.parametrize('env', [None, '0', '1'])
def test_explicit_argument_beats_the_env(monkeypatch, env):
    if env is not None:
        monkeypatch.setenv('MEDIAPIPE_USE_GPU', env)
    assert delegate.gpu_requested(True) is True
    assert delegate.gpu_requested(False) is False


# ---------------------------------------------------------------------------
# create_detector
# ---------------------------------------------------------------------------

def test_default_never_touches_the_gpu():
    build = _recorder()
    result = delegate.create_detector(build, what='Pose')

    assert result == 'detector(%s)' % CPU
    assert build.seen == [CPU], 'GPU was attempted despite being opt-in'


def test_use_gpu_false_never_touches_the_gpu(monkeypatch):
    monkeypatch.setenv('MEDIAPIPE_USE_GPU', '1')   # even with the env set
    build = _recorder()

    delegate.create_detector(build, use_gpu=False, what='Pose')

    assert build.seen == [CPU]


def test_gpu_tried_first_when_requested():
    build = _recorder()
    result = delegate.create_detector(build, use_gpu=True, what='Pose')

    assert result == 'detector(%s)' % GPU
    assert build.seen == [GPU]


def test_falls_back_to_cpu_when_gpu_build_raises(caplog):
    build = _recorder(fail_on=(GPU,))

    with caplog.at_level('WARNING'):
        result = delegate.create_detector(build, use_gpu=True, what='Pose')

    assert result == 'detector(%s)' % CPU
    assert build.seen == [GPU, CPU]
    assert 'GPU delegate unavailable' in caplog.text
    assert 'RuntimeError' in caplog.text


def test_gpu_failure_is_not_retried_for_that_detector_kind():
    """A GPU-less host must not re-pay a failed GL init on every creation."""
    build = _recorder(fail_on=(GPU,))

    for _ in range(4):
        delegate.create_detector(build, use_gpu=True, what='Pose')

    assert build.seen == [GPU, CPU, CPU, CPU, CPU], (
        'GPU retried after a known failure: %r' % (build.seen,))


def test_the_cache_is_per_detector_kind():
    build = _recorder(fail_on=(GPU,))

    delegate.create_detector(build, use_gpu=True, what='Pose')
    delegate.create_detector(build, use_gpu=True, what='Segmenter')

    # Each kind gets its own first attempt; neither poisons the other.
    assert build.seen == [GPU, CPU, GPU, CPU]


def test_a_failure_does_not_suppress_the_cpu_log(caplog):
    """The old code added `what` to the log-dedup set on failure, which meant the
    CPU line it fell back to was never emitted. Keying on (kind, delegate) fixes
    that -- the log must state what is actually in use."""
    build = _recorder(fail_on=(GPU,))

    with caplog.at_level('INFO'):
        delegate.create_detector(build, use_gpu=True, what='Pose')

    assert 'using CPU delegate' in caplog.text


def test_delegate_log_is_deduped_per_kind_and_delegate(caplog):
    build = _recorder()

    with caplog.at_level('INFO'):
        for _ in range(3):
            delegate.create_detector(build, what='Pose')

    assert caplog.text.count('using CPU delegate') == 1


def test_log_reports_a_later_switch_rather_than_hiding_it(caplog):
    """Same detector kind, CPU then GPU: both must be reported. The old dedup
    keyed on the kind alone, so the second line was swallowed and the log
    asserted CPU while GPU was in use."""
    build = _recorder()

    with caplog.at_level('INFO'):
        delegate.create_detector(build, use_gpu=False, what='Pose')
        delegate.create_detector(build, use_gpu=True, what='Pose')

    assert 'using CPU delegate' in caplog.text
    assert 'using GPU delegate' in caplog.text


def test_cpu_failure_propagates():
    """Only the GPU attempt is forgiving. If CPU can't be built, something is
    genuinely wrong and the caller must hear about it."""
    build = _recorder(fail_on=(GPU, CPU))

    with pytest.raises(RuntimeError):
        delegate.create_detector(build, use_gpu=True, what='Pose')


def test_mediapipe_is_not_imported_at_module_scope():
    """The repo convention (optional deps lazy-imported) is what makes this test
    file possible at all -- pin it so a top-level import can't creep back."""
    source = _PATH.read_text()
    top_level = [ln for ln in source.splitlines()
                 if ln.startswith(('import ', 'from ')) and 'mediapipe' in ln]
    assert not top_level, 'mediapipe imported at module scope: %s' % top_level


def test_public_api_is_declared():
    assert delegate.__all__ == ['gpu_requested', 'create_detector']
